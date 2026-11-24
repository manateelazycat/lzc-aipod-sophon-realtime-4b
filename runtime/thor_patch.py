"""Thor (Blackwell sm_110a) runtime acceleration hooks for Sophon-Realtime (32B & 4B).

Activated selectively via environment variables (all disabled by default):
  SOPHON_THOR_THINKER_MARLIN=w4_g128|w4_g64|w8  : On-the-fly Marlin W4/W8 for Stage-0 CompressedTensors FP8 Thinker (32B)
  SOPHON_THOR_TALKER_MARLIN=w4_g64|w8           : On-the-fly Marlin W4/W8 for Stage-1 BF16 Talker (and 4B Thinker via BF16 hook)
  SOPHON_THOR_BF16_LINEAR_MARLIN=w4_g128|w4_g64|w8 : On-the-fly Marlin W4/W8 for UnquantizedLinearMethod (4B Thinker / Talker)
  SOPHON_THOR_FLOW_SKIP_ZERO_CFG=1              : Skip dead unconditional half in ConditionalCFM.solve_euler when cfg_rate==0
  SOPHON_THOR_FLOW_BF16=1                       : Run CosyVoice3 Flow DiT estimator in native BF16 on Blackwell Tensor Cores
  SOPHON_THOR_FLOW_CG=1                         : Per-shape CUDA Graph cache for CosyVoice3 Flow DiT estimator
"""
import os
import sys
import torch
from torch.nn import Parameter

_WS_CACHE = {}


def _get_ws(device):
    ws = _WS_CACHE.get(device)
    if ws is None:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
        ws = marlin_make_workspace_new(device)
        _WS_CACHE[device] = ws
    return ws


def _pack_w_marlin(w_f32: torch.Tensor, mode: str):
    """Pack [N, K] float32 weight matrix into vLLM 0.30.0 Marlin layout on CUDA (vectorized, low-mem)."""
    import vllm._custom_ops as ops
    from vllm.scalar_type import scalar_types
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales

    n, k = w_f32.shape
    if mode in ("w4_g128", "w4_g64"):
        g = 128 if mode == "w4_g128" else 64
        wf = w_f32.view(n, k // g, g)
        sc = wf.abs().amax(dim=-1).clamp_(min=1e-7).div_(7.0)
        wf.div_(sc[..., None]).round_().clamp_(-8.0, 7.0).add_(8.0)
        qu = wf.to(torch.uint8).view(n, k // 2, 2)
        del wf
        packed_u8 = qu[..., 0] | (qu[..., 1] << 4)
        del qu
        qw = packed_u8.contiguous().view(torch.int32).t().contiguous()
        del packed_u8
        q_marlin = ops.gptq_marlin_repack(qw, k, n, 4)
        sc_marlin = marlin_permute_scales(sc.t().to(torch.bfloat16).contiguous(), k, n, g)
        del qw, sc
        return q_marlin, sc_marlin, scalar_types.uint4b8
    elif mode == "w8":
        sc = w_f32.abs().amax(dim=-1).clamp_(min=1e-7).div_(127.0)
        w_f32.div_(sc[:, None]).round_().clamp_(-128.0, 127.0).add_(128.0)
        qw = w_f32.to(torch.uint8).contiguous().view(torch.int32).t().contiguous()
        q_marlin = ops.gptq_marlin_repack(qw, k, n, 8)
        sc_marlin = marlin_permute_scales(sc[None, :].to(torch.bfloat16).contiguous(), k, n, -1)
        del qw, sc
        return q_marlin, sc_marlin, scalar_types.uint8b128
    else:
        raise ValueError(f"Unknown Marlin mode: {mode}")


def _install_thinker_fp8_to_marlin(mode: str):
    from compressed_tensors.quantization import QuantizationStrategy
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
        CompressedTensorsW8A8Fp8,
    )

    if getattr(CompressedTensorsW8A8Fp8, "_thor_marlin_patched", False):
        return
    CompressedTensorsW8A8Fp8._thor_marlin_patched = True

    orig_create = CompressedTensorsW8A8Fp8.create_weights
    orig_process = CompressedTensorsW8A8Fp8.process_weights_after_loading
    orig_apply = CompressedTensorsW8A8Fp8.apply_weights

    g_div = 128 if mode == "w4_g128" else (64 if mode == "w4_g64" else 16)
    packed_count = [0]

    def create_weights_patched(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype, weight_loader, **kwargs):
        orig_create(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype, weight_loader, **kwargs)
        if self.strategy == QuantizationStrategy.CHANNEL:
            n_out = sum(output_partition_sizes)
            if (input_size_per_partition % g_div == 0) and (n_out % 64 == 0):
                if hasattr(layer, "input_quant_key"):
                    layer.input_quant_key = None

    def process_weights_patched(self, layer):
        if self.strategy == QuantizationStrategy.CHANNEL:
            w = layer.weight.data
            n, k = w.shape
            if (k % g_div == 0) and (n % 64 == 0):
                sc = layer.weight_scale.data
                if sc.ndim == 1:
                    sc = sc[:, None]
                w_f32 = w.float() * sc.float()
                q_marlin, sc_marlin, stype = _pack_w_marlin(w_f32, mode)
                del w_f32
                layer.weight = Parameter(q_marlin, requires_grad=False)
                layer.weight_scale = Parameter(sc_marlin, requires_grad=False)
                layer.input_scale = None
                layer.register_buffer("_thor_ws", _get_ws(q_marlin.device), persistent=False)
                layer._thor_marlin = (n, k, stype)
                packed_count[0] += 1
                if packed_count[0] in (1, 64, 128, 256):
                    print(f"[thor_patch] Packed {packed_count[0]} Thinker FP8->Marlin ({mode}) layers (last {n}x{k})", flush=True)
                return
        orig_process(self, layer)

    def apply_weights_patched(self, layer, x, bias=None):
        info = getattr(layer, "_thor_marlin", None)
        if info is not None:
            n, k, stype = info
            orig_shape = x.shape
            x2d = x.reshape(-1, k)
            if x2d.dtype != torch.bfloat16:
                x2d = x2d.to(torch.bfloat16)
            if not x2d.is_contiguous():
                x2d = x2d.contiguous()
            b_arg = bias.to(torch.bfloat16) if (bias is not None and bias.dtype != torch.bfloat16) else bias
            out = ops.marlin_gemm(
                x2d, None, layer.weight, None, layer.weight_scale,
                None, None, None, layer._thor_ws, stype,
                x2d.shape[0], n, k, False, False, False,
            )
            if b_arg is not None:
                out = out + b_arg
            return out.reshape(*orig_shape[:-1], n)
        return orig_apply(self, layer, x, bias)

    CompressedTensorsW8A8Fp8.create_weights = create_weights_patched
    CompressedTensorsW8A8Fp8.process_weights_after_loading = process_weights_patched
    CompressedTensorsW8A8Fp8.apply_weights = apply_weights_patched
    print(f"[thor_patch] Installed Thinker FP8 -> Marlin ({mode}) hook", flush=True)


def _install_bf16_linear_to_marlin(mode: str, cp_mode: str = "", min_k: int = 512, min_n: int = 512):
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    if getattr(UnquantizedLinearMethod, "_thor_marlin_patched", False):
        return
    UnquantizedLinearMethod._thor_marlin_patched = True

    orig_process = UnquantizedLinearMethod.process_weights_after_loading
    orig_apply = UnquantizedLinearMethod.apply
    packed_count = [0]
    skip_substrings = (
        "audio_tower", "audio_encoder", "vision_tower", "visual", "multi_modal",
        "projector", "lm_head", "codec_head", "text_projection", "llm_decoder",
    )

    talker_override = os.environ.get("SOPHON_THOR_TALKER_MARLIN", "").strip().lower()

    def _mode_for_layer(layer):
        prefix = str(getattr(layer, "prefix", "") or "")
        for bad in skip_substrings:
            if bad in prefix:
                return ""
        if cp_mode and "code_predictor" in prefix:
            return cp_mode
        if talker_override in ("w4_g128", "w4_g64", "w8", "none") and prefix.startswith("model.layers."):
            return "" if talker_override == "none" else talker_override
        return mode

    def process_patched(self, layer):
        orig_process(self, layer)
        m = _mode_for_layer(layer)
        if not m or m == "none":
            return
        g_div = 128 if m == "w4_g128" else (64 if m == "w4_g64" else 16)
        w = getattr(layer, "weight", None)
        if w is not None and w.is_cuda and w.ndim == 2 and w.dtype in (torch.bfloat16, torch.float16):
            n, k = w.shape
            if k >= min_k and n >= min_n and n <= 32768 and (k % g_div == 0) and (n % 64 == 0):
                q_marlin, sc_marlin, stype = _pack_w_marlin(w.data.float(), m)
                layer.weight = Parameter(q_marlin, requires_grad=False)
                layer.weight_scale = Parameter(sc_marlin, requires_grad=False)
                layer.register_buffer("_thor_ws", _get_ws(q_marlin.device), persistent=False)
                layer._thor_marlin = (n, k, stype)
                packed_count[0] += 1
                if packed_count[0] in (1, 32, 64, 128, 256):
                    print(f"[thor_patch] Packed {packed_count[0]} BF16->Marlin ({m}) layers (last {getattr(layer, 'prefix', '')} {n}x{k})", flush=True)

    def apply_patched(self, layer, x, bias=None):
        info = getattr(layer, "_thor_marlin", None)
        if info is not None:
            n, k, stype = info
            orig_shape = x.shape
            x2d = x.reshape(-1, k)
            if x2d.dtype != torch.bfloat16:
                x2d = x2d.to(torch.bfloat16)
            if not x2d.is_contiguous():
                x2d = x2d.contiguous()
            b_arg = bias.to(torch.bfloat16) if (bias is not None and bias.dtype != torch.bfloat16) else bias
            out = ops.marlin_gemm(
                x2d, None, layer.weight, None, layer.weight_scale,
                None, None, None, layer._thor_ws, stype,
                x2d.shape[0], n, k, False, False, False,
            )
            if b_arg is not None:
                out = out + b_arg
            return out.reshape(*orig_shape[:-1], n)
        return orig_apply(self, layer, x, bias)

    UnquantizedLinearMethod.process_weights_after_loading = process_patched
    UnquantizedLinearMethod.apply = apply_patched
    print(f"[thor_patch] Installed BF16 UnquantizedLinear -> Marlin ({mode}) hook", flush=True)


def _install_flow_fast(skip_zero_cfg: bool, flow_bf16: bool, flow_cg: bool):
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import ConditionalCFM

    if getattr(ConditionalCFM, "_thor_flow_patched", False):
        return
    ConditionalCFM._thor_flow_patched = True

    orig_solve = ConditionalCFM.solve_euler
    orig_fwd_est = ConditionalCFM.forward_estimator

    def solve_euler_fast(self, x, t_span, mu, mask, spks, cond):
        if skip_zero_cfg and abs(float(getattr(self, "inference_cfg_rate", 0.7))) < 1e-6:
            t = t_span[0].unsqueeze(0)
            dt = t_span[1] - t_span[0]
            batch_size = int(x.size(0))
            est_dtype = spks.dtype if spks is not None else x.dtype
            x_cur = x.to(est_dtype)
            mask_in = mask.to(est_dtype)
            mu_in = mu.to(est_dtype)
            spks_in = spks.to(est_dtype) if spks is not None else torch.zeros([batch_size, 80], device=x.device, dtype=est_dtype)
            cond_in = cond.to(est_dtype) if cond is not None else torch.zeros([batch_size, 80, x.size(2)], device=x.device, dtype=est_dtype)
            t_in = torch.empty([batch_size], device=x.device, dtype=est_dtype)
            for step in range(1, len(t_span)):
                t_in.fill_(t.item())
                dphi_dt = self.forward_estimator(x_cur, mask_in, mu_in, t_in, spks_in, cond_in)
                x_cur = x_cur + dt * dphi_dt
                t = t + dt
                if step < len(t_span) - 1:
                    dt = t_span[step + 1] - t
            return x_cur.float()
        return orig_solve(self, x, t_span, mu, mask, spks, cond)

    def forward_estimator_fast(self, x, mask, mu, t, spks, cond):
        if not isinstance(self.estimator, torch.nn.Module):
            return orig_fwd_est(self, x, mask, mu, t, spks, cond)
        out_dtype = x.dtype
        if flow_bf16:
            if not getattr(self.estimator, "_thor_bf16_converted", False):
                self.estimator.to(dtype=torch.bfloat16)
                self.estimator._thor_bf16_converted = True
                print("[thor_patch] Converted CosyVoice3 Flow DiT estimator to native BF16!", flush=True)
            x = x.to(torch.bfloat16)
            mask = mask.to(torch.bfloat16)
            mu = mu.to(torch.bfloat16)
            t = t.to(torch.bfloat16)
            spks = spks.to(torch.bfloat16)
            cond = cond.to(torch.bfloat16)
        if flow_cg:
            cache = getattr(self, "_thor_est_graphs", None)
            if cache is None:
                cache = {}
                self._thor_est_graphs = cache
            key = (tuple(x.shape), x.dtype)
            entry = cache.get(key)
            if entry is None and len(cache) < 32:
                sx = x.contiguous().clone()
                smask = mask.contiguous().clone()
                smu = mu.contiguous().clone()
                st = t.contiguous().clone()
                sspks = spks.contiguous().clone()
                scond = cond.contiguous().clone()
                try:
                    # Warmup + capture
                    for _ in range(2):
                        self.estimator(sx, smask, smu, st, sspks, scond)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        sout = self.estimator(sx, smask, smu, st, sspks, scond)
                    torch.cuda.synchronize()
                    cache[key] = (g, sx, smask, smu, st, sspks, scond, sout)
                    entry = cache[key]
                except Exception as exc:
                    print(f"[thor_patch] Flow CG capture failed for {key}: {exc}", flush=True)
                    cache[key] = False
                    entry = False
            if entry and entry is not False:
                g, sx, smask, smu, st, sspks, scond, sout = entry
                sx.copy_(x); smask.copy_(mask); smu.copy_(mu); st.copy_(t); sspks.copy_(spks); scond.copy_(cond)
                g.replay()
                return sout.to(out_dtype)
        return self.estimator(x, mask, mu, t, spks, cond).to(out_dtype)

    ConditionalCFM.solve_euler = solve_euler_fast
    ConditionalCFM.forward_estimator = forward_estimator_fast
    print(f"[thor_patch] Installed Flow fast hook (skip_zero_cfg={skip_zero_cfg}, bf16={flow_bf16}, cg={flow_cg})", flush=True)


_CP_TRITON_ATTN_KERNEL = None


def _get_cp_triton_attn_kernel():
    global _CP_TRITON_ATTN_KERNEL
    if _CP_TRITON_ATTN_KERNEL is not None:
        return _CP_TRITON_ATTN_KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def _cp_decode_attn_kernel(
        QKV, QNW, KNW, COS, SIN, KC, VC, OUT,
        stride_qkv_b, stride_kc_b, stride_kc_h, stride_kc_t, stride_out_b,
        SCALE: tl.constexpr, EPS: tl.constexpr,
        POS: tl.constexpr, KV_LEN: tl.constexpr,
        HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
    ):
        b = tl.program_id(0)
        hkv = tl.program_id(1)
        G: tl.constexpr = HQ // HKV
        HALF: tl.constexpr = D // 2
        offs_half = tl.arange(0, HALF)
        offs_d = tl.arange(0, D)

        c1 = tl.load(COS + POS * D + offs_half).to(tl.float32)
        c2 = tl.load(COS + POS * D + HALF + offs_half).to(tl.float32)
        s1 = tl.load(SIN + POS * D + offs_half).to(tl.float32)
        s2 = tl.load(SIN + POS * D + HALF + offs_half).to(tl.float32)

        qkv_base = QKV + b * stride_qkv_b
        k_off = HQ * D + hkv * D
        v_off = (HQ + HKV) * D + hkv * D
        k1 = tl.load(qkv_base + k_off + offs_half).to(tl.float32)
        k2 = tl.load(qkv_base + k_off + HALF + offs_half).to(tl.float32)
        k_var = (tl.sum(k1 * k1) + tl.sum(k2 * k2)) / D
        k_inv = tl.rsqrt(k_var + EPS)
        kw1 = tl.load(KNW + offs_half).to(tl.float32)
        kw2 = tl.load(KNW + HALF + offs_half).to(tl.float32)
        k1 = k1 * k_inv * kw1
        k2 = k2 * k_inv * kw2
        kr1 = k1 * c1 - k2 * s1
        kr2 = k2 * c2 + k1 * s2

        kv_base = b * stride_kc_b + hkv * stride_kc_h
        # Load strictly previous KV entries (t < POS) -> zero intra-CTA store-to-load race!
        t_idx = tl.arange(0, 16)
        prev_mask = t_idx < POS
        k_lo = tl.load(KC + kv_base + t_idx[:, None] * stride_kc_t + offs_half[None, :], mask=prev_mask[:, None], other=0.0).to(tl.float32)
        k_hi = tl.load(KC + kv_base + t_idx[:, None] * stride_kc_t + HALF + offs_half[None, :], mask=prev_mask[:, None], other=0.0).to(tl.float32)
        V_prev = tl.load(VC + kv_base + t_idx[:, None] * stride_kc_t + offs_d[None, :], mask=prev_mask[:, None], other=0.0).to(tl.float32)

        # Store current step K & V for subsequent steps; keep exact values in registers for current step!
        tl.store(KC + kv_base + POS * stride_kc_t + offs_half, kr1.to(tl.bfloat16))
        tl.store(KC + kv_base + POS * stride_kc_t + HALF + offs_half, kr2.to(tl.bfloat16))
        v_new_bf16 = tl.load(qkv_base + v_off + offs_d)
        tl.store(VC + kv_base + POS * stride_kc_t + offs_d, v_new_bf16)
        v_new = v_new_bf16.to(tl.float32)

        qw1 = tl.load(QNW + offs_half).to(tl.float32)
        qw2 = tl.load(QNW + HALF + offs_half).to(tl.float32)

        for g in tl.static_range(G):
            hq = hkv * G + g
            q_off = hq * D
            q1 = tl.load(qkv_base + q_off + offs_half).to(tl.float32)
            q2 = tl.load(qkv_base + q_off + HALF + offs_half).to(tl.float32)
            q_var = (tl.sum(q1 * q1) + tl.sum(q2 * q2)) / D
            q_inv = tl.rsqrt(q_var + EPS)
            q1 = q1 * q_inv * qw1
            q2 = q2 * q_inv * qw2
            qr1 = q1 * c1 - q2 * s1
            qr2 = q2 * c2 + q1 * s2
            scores_prev = (tl.sum(k_lo * qr1[None, :], axis=1) + tl.sum(k_hi * qr2[None, :], axis=1)) * SCALE
            scores_prev = tl.where(prev_mask, scores_prev, -1e9)
            score_cur = (tl.sum(kr1 * qr1) + tl.sum(kr2 * qr2)) * SCALE
            m = tl.maximum(tl.max(scores_prev, axis=0), score_cur)
            p_prev = tl.where(prev_mask, tl.exp(scores_prev - m), 0.0)
            p_cur = tl.exp(score_cur - m)
            denom = tl.sum(p_prev, axis=0) + p_cur
            out_vec = (tl.sum(V_prev * p_prev[:, None], axis=0) + v_new * p_cur) / denom
            tl.store(OUT + b * stride_out_b + hq * D + offs_d, out_vec.to(tl.bfloat16))

    _CP_TRITON_ATTN_KERNEL = _cp_decode_attn_kernel
    return _CP_TRITON_ATTN_KERNEL


def _install_cp_fast(cp_mode: str, fused_rmsnorm: bool, cp_kvcache: bool):
    import torch.nn.functional as F
    import vllm._custom_ops as ops
    from vllm_omni.model_executor.models.common import qwen3_code_predictor as qcp

    if fused_rmsnorm and not getattr(qcp._RMSNorm, "_thor_rmsnorm_patched", False):
        qcp._RMSNorm._thor_rmsnorm_patched = True

        def rmsnorm_cuda_fast(self, hidden_states: torch.Tensor) -> torch.Tensor:
            orig_shape = hidden_states.shape
            x2d = hidden_states.reshape(-1, orig_shape[-1]).contiguous()
            out = torch.empty_like(x2d)
            w = self.weight if self.weight.dtype == x2d.dtype else self.weight.to(x2d.dtype)
            ops.rms_norm(out, x2d, w, self.variance_epsilon)
            return out.reshape(orig_shape)

        qcp._RMSNorm.forward_cuda = rmsnorm_cuda_fast
        print("[thor_patch] Installed CodePredictor fused vLLM rms_norm hook", flush=True)

    if cp_mode in ("w4_g128", "w4_g64", "w8") and not getattr(qcp.CodePredictorWrapper, "_thor_cp_marlin_patched", False):
        qcp.CodePredictorWrapper._thor_cp_marlin_patched = True
        orig_setup = qcp.CodePredictorWrapper._setup_compile
        g_div = 128 if cp_mode == "w4_g128" else (64 if cp_mode == "w4_g64" else 16)

        class _MarlinLinear(torch.nn.Module):
            def __init__(self, linear: torch.nn.Linear, mode: str):
                super().__init__()
                w = linear.weight.data
                n, k = w.shape
                q_marlin, sc_marlin, stype = _pack_w_marlin(w.float(), mode)
                self.weight = Parameter(q_marlin, requires_grad=False)
                self.weight_scale = Parameter(sc_marlin, requires_grad=False)
                if linear.bias is not None:
                    self.bias = Parameter(linear.bias.data.to(torch.bfloat16).contiguous(), requires_grad=False)
                else:
                    self.bias = None
                self.register_buffer("_thor_ws", _get_ws(q_marlin.device), persistent=False)
                self._n = n
                self._k = k
                self._stype = stype

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                orig_shape = x.shape
                x2d = x.reshape(-1, self._k)
                if x2d.dtype != torch.bfloat16:
                    x2d = x2d.to(torch.bfloat16)
                if not x2d.is_contiguous():
                    x2d = x2d.contiguous()
                out = ops.marlin_gemm(
                    x2d, None, self.weight, None, self.weight_scale,
                    None, None, None, self._thor_ws, self._stype,
                    x2d.shape[0], self._n, self._k, False, False, False,
                )
                if self.bias is not None:
                    out = out + self.bias
                return out.reshape(*orig_shape[:-1], self._n)

        def _pack_cp_layers(wrapper):
            if getattr(wrapper.model, "_thor_marlin_packed", False):
                return
            wrapper.model._thor_marlin_packed = True
            cnt = 0
            for layer in wrapper.model.layers:
                for parent, attr in (
                    (layer.self_attn, "qkv_proj"),
                    (layer.self_attn, "o_proj"),
                    (layer.mlp, "gate_up_proj"),
                    (layer.mlp, "down_proj"),
                ):
                    lin = getattr(parent, attr, None)
                    if isinstance(lin, torch.nn.Linear) and lin.weight.is_cuda:
                        n, k = lin.weight.shape
                        if (k % g_div == 0) and (n % 64 == 0):
                            setattr(parent, attr, _MarlinLinear(lin, cp_mode))
                            cnt += 1
            head_cnt = 0
            if os.environ.get("SOPHON_THOR_CP_HEAD_W8", "0") == "1":
                for i, head in enumerate(wrapper.lm_head):
                    if isinstance(head, torch.nn.Linear) and head.weight.is_cuda:
                        n, k = head.weight.shape
                        if (k % 16 == 0) and (n % 64 == 0):
                            wrapper.lm_head[i] = _MarlinLinear(head, "w8")
                            head_cnt += 1
                proj = getattr(wrapper, "small_to_mtp_projection", None)
                if isinstance(proj, torch.nn.Linear) and proj.weight.is_cuda:
                    n, k = proj.weight.shape
                    if (k % 16 == 0) and (n % 64 == 0):
                        wrapper.small_to_mtp_projection = _MarlinLinear(proj, "w8")
                        head_cnt += 1
            print(f"[thor_patch] Packed {cnt} CodePredictor nn.Linear -> Marlin ({cp_mode}) + {head_cnt} heads -> Marlin (w8)", flush=True)

        def setup_compile_patched(self):
            if self._compiled_model_fwd is None:
                _pack_cp_layers(self)
                if cp_kvcache:
                    # Skip 17-token re-prefill Inductor compile when KV-cache path is active!
                    self._model_dtype = self.model.norm.weight.dtype
                    self._lm_heads_list = list(self.lm_head)
                    self._codec_embeds_list = list(self.model.codec_embedding)
                    self._compiled_model_fwd = self.model.forward
                    return
            return orig_setup(self)

        qcp.CodePredictorWrapper._setup_compile = setup_compile_patched
        print(f"[thor_patch] Installed CodePredictor Marlin ({cp_mode}) hook", flush=True)

    if cp_kvcache and not getattr(qcp.CodePredictorWrapper, "_thor_cp_kvcache_patched", False):
        qcp.CodePredictorWrapper._thor_cp_kvcache_patched = True
        _rotate_half = qcp._rotate_half

        use_triton_attn = os.environ.get("SOPHON_THOR_CP_TRITON_ATTN", "0") == "1"
        triton_attn_kern = _get_cp_triton_attn_kernel() if use_triton_attn else None

        def _ensure_kv_cache(self, bsz: int, device: torch.device, dtype: torch.dtype):
            cache = getattr(self, "_thor_kv_caches", None)
            num_layers = len(self.model.layers)
            attn0 = self.model.layers[0].self_attn
            mlp0 = self.model.layers[0].mlp
            max_seq = self._num_groups + 1
            if (
                cache is None
                or cache[0][0].shape[0] < bsz
                or cache[0][0].device != device
                or cache[0][0].dtype != dtype
            ):
                cap_b = max(bsz, 8)
                cache = [
                    (
                        torch.empty((cap_b, attn0.num_kv_heads, max_seq, attn0.head_dim), device=device, dtype=dtype),
                        torch.empty((cap_b, attn0.num_kv_heads, max_seq, attn0.head_dim), device=device, dtype=dtype),
                    )
                    for _ in range(num_layers)
                ]
                self._thor_kv_caches = cache
                self._thor_pos_table = torch.arange(max_seq, device=device, dtype=torch.long)
                self._thor_attn_out_buf = torch.empty((cap_b, attn0.num_heads * attn0.head_dim), device=device, dtype=dtype)
                self._thor_gate_buf = torch.empty((cap_b, 2, mlp0._intermediate_size), device=device, dtype=dtype)
                self._thor_cos_bf16 = self.model.rotary_emb.cos_cached.to(device=device, dtype=dtype).contiguous()
                self._thor_sin_bf16 = self.model.rotary_emb.sin_cached.to(device=device, dtype=dtype).contiguous()
            return cache

        def _kv_step_layers(self, x: torch.Tensor, start_pos: int, seq_len: int, kv_caches):
            bsz = x.shape[0]
            end_pos = start_pos + seq_len
            use_fused = (triton_attn_kern is not None and seq_len == 1 and x.dtype == torch.bfloat16)
            if not use_fused:
                pos = self._thor_pos_table[start_pos : end_pos].unsqueeze(0).expand(bsz, -1)
                cos, sin = self.model.rotary_emb(x, pos)
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
            h = x
            for idx, layer in enumerate(self.model.layers):
                residual = h
                h = layer.input_layernorm(h)
                attn = layer.self_attn
                qkv = attn.qkv_proj(h)
                kc, vc = kv_caches[idx]
                if use_fused:
                    qkv2d = qkv.view(bsz, -1)
                    aout = self._thor_attn_out_buf[:bsz]
                    triton_attn_kern[(bsz, attn.num_kv_heads)](
                        qkv2d, attn.q_norm.weight, attn.k_norm.weight,
                        self._thor_cos_bf16, self._thor_sin_bf16, kc, vc, aout,
                        qkv2d.stride(0), kc.stride(0), kc.stride(1), kc.stride(2), aout.stride(0),
                        attn.scaling, attn.q_norm.variance_epsilon,
                        start_pos, end_pos, attn.num_heads, attn.num_kv_heads, attn.head_dim,
                    )
                    h = attn.o_proj(aout.unsqueeze(1))
                else:
                    q_raw, k_raw, v_raw = attn._split_qkv(qkv)
                    q = attn.q_norm(q_raw.view(bsz, seq_len, attn.num_heads, attn.head_dim)).transpose(1, 2)
                    k = attn.k_norm(k_raw.view(bsz, seq_len, attn.num_kv_heads, attn.head_dim)).transpose(1, 2)
                    v = v_raw.view(bsz, seq_len, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
                    q = (q * cos) + (_rotate_half(q) * sin)
                    k = (k * cos) + (_rotate_half(k) * sin)
                    kc[:bsz, :, start_pos:end_pos, :].copy_(k)
                    vc[:bsz, :, start_pos:end_pos, :].copy_(v)
                    k_all = kc[:bsz, :, :end_pos, :]
                    v_all = vc[:bsz, :, :end_pos, :]
                    attn_out = F.scaled_dot_product_attention(
                        q, k_all, v_all,
                        scale=attn.scaling,
                        is_causal=(seq_len > 1),
                        enable_gqa=attn.is_gqa,
                    )
                    attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
                    h = attn.o_proj(attn_out)
                h = residual + h
                residual = h
                h = layer.post_attention_layernorm(h)
                mlp = layer.mlp
                gu = mlp.gate_up_proj(h)
                if hasattr(torch.ops._C, "silu_and_mul"):
                    gate_out = self._thor_gate_buf[:bsz, :seq_len, :]
                    torch.ops._C.silu_and_mul(gate_out.reshape(-1, mlp._intermediate_size), gu.reshape(-1, 2 * mlp._intermediate_size))
                else:
                    gate, up = gu.split([mlp._intermediate_size, mlp._intermediate_size], dim=-1)
                    gate_out = F.silu(gate) * up
                h = mlp.down_proj(gate_out)
                h = residual + h
            return self.model.norm(h)

        @torch.inference_mode()
        def forward_kvcache(
            self,
            layer0_code: torch.Tensor,
            layer0_embed: torch.Tensor,
            last_talker_hidden: torch.Tensor,
            do_sample: bool = True,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 1.0,
            generator=None,
            generators=None,
            sample_uniforms=None,
        ):
            bsz = int(layer0_code.shape[0])
            num_groups = self._num_groups
            sample_generator = generators if generators is not None else generator
            device = layer0_code.device

            self._setup_compile()
            dtype = self._model_dtype
            kv_caches = _ensure_kv_cache(self, bsz, device, dtype)

            projection = self.small_to_mtp_projection
            lm_heads = self._lm_heads_list
            codec_embeds = self._codec_embeds_list

            stored_mode = self._wrapper_config.sampling_mode == "stored"
            if stored_mode:
                s_top_k = self._top_k
                s_top_p = self._top_p
            else:
                use_sampling = do_sample and temperature > 0
                inv_temperature = 1.0 / max(temperature, 1e-6) if use_sampling else 0.0

            if self._wrapper_config.return_proj_buf:
                self._ensure_buffers(device, dtype, bsz)
                proj_buf = self._proj_buf
                all_codes = torch.empty(bsz, num_groups, 1, dtype=torch.int64, device=device)
                all_codes[:, 0] = layer0_code.reshape(bsz, -1)[:, :1]
            else:
                proj_buf = None
                all_codes = torch.empty(bsz, num_groups, dtype=torch.long, device=device)
                all_codes[:, 0] = layer0_code.reshape(bsz)

            tok0 = projection(last_talker_hidden.reshape(bsz, 1, -1).to(dtype))
            tok1 = projection(layer0_embed.reshape(bsz, 1, -1).to(dtype))
            x_init = torch.cat([tok0, tok1], dim=1)
            if proj_buf is not None:
                proj_buf[:bsz, 0:2, :].copy_(x_init)

            for step in range(1, num_groups):
                if step == 1:
                    h_out = _kv_step_layers(self, x_init, 0, 2, kv_caches)
                    last_h = h_out[:, 1, :]
                else:
                    h_out = _kv_step_layers(self, next_tok, step, 1, kv_caches)
                    last_h = h_out[:, 0, :]

                logits = lm_heads[step - 1](last_h)

                if stored_mode:
                    if s_top_k > 0:
                        topk_vals, _ = logits.topk(s_top_k, dim=-1)
                        logits = logits.masked_fill(logits < topk_vals[:, -1:], float("-inf"))
                    if s_top_p < 1.0:
                        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                        sorted_probs = F.softmax(sorted_logits, dim=-1, dtype=torch.float32)
                        cumulative_probs = sorted_probs.cumsum(dim=-1)
                        remove_mask = (cumulative_probs - sorted_probs) >= s_top_p
                        sorted_logits[remove_mask] = float("-inf")
                        logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
                    step_uniforms = sample_uniforms[:, step - 1, :] if sample_uniforms is not None else None
                    code = self._sample_codes_gumbel(logits, generator=sample_generator, uniforms=step_uniforms)
                else:
                    if use_sampling:
                        scaled = logits * inv_temperature
                        if top_k > 0:
                            topk_vals, _ = scaled.topk(top_k, dim=-1)
                            scaled = scaled.masked_fill(scaled < topk_vals[:, -1:], float("-inf"))
                        step_uniforms = sample_uniforms[:, step - 1, :] if sample_uniforms is not None else None
                        code = self._sample_codes_gumbel(scaled, generator=sample_generator, uniforms=step_uniforms)
                    else:
                        code = logits.argmax(dim=-1, keepdim=True)

                if self._wrapper_config.return_proj_buf:
                    all_codes[:, step] = code
                else:
                    all_codes[:, step] = code.reshape(bsz)

                if step < num_groups - 1 or self._wrapper_config.return_proj_buf:
                    new_embed = codec_embeds[step - 1](code)
                    next_tok = projection(new_embed.reshape(bsz, 1, -1))
                    if proj_buf is not None:
                        proj_buf[:bsz, step + 1 : step + 2, :].copy_(next_tok)

            if self._wrapper_config.return_proj_buf:
                return all_codes, proj_buf[:bsz].clone()
            return all_codes

        qcp.CodePredictorWrapper.forward = forward_kvcache
        print("[thor_patch] Installed CodePredictor static-KV-cache 1-token decoder hook", flush=True)


def install():
    thinker_mode = os.environ.get("SOPHON_THOR_THINKER_MARLIN", "").strip().lower()
    if thinker_mode in ("w4_g128", "w4_g64", "w8"):
        try:
            _install_thinker_fp8_to_marlin(thinker_mode)
        except Exception as exc:
            print(f"[thor_patch] Thinker Marlin hook failed: {exc}", file=sys.stderr, flush=True)

    bf16_mode = os.environ.get("SOPHON_THOR_BF16_LINEAR_MARLIN", "").strip().lower()
    if bf16_mode in ("w4_g128", "w4_g64", "w8"):
        try:
            _install_bf16_linear_to_marlin(bf16_mode)
        except Exception as exc:
            print(f"[thor_patch] BF16 Linear Marlin hook failed: {exc}", file=sys.stderr, flush=True)

    cp_mode = os.environ.get("SOPHON_THOR_CP_MARLIN", "").strip().lower()
    cp_rmsnorm = os.environ.get("SOPHON_THOR_CP_RMSNORM", "0") == "1"
    cp_kvcache = os.environ.get("SOPHON_THOR_CP_KVCACHE", "0") == "1"
    if cp_mode in ("w4_g128", "w4_g64", "w8") or cp_rmsnorm or cp_kvcache:
        try:
            _install_cp_fast(cp_mode, cp_rmsnorm, cp_kvcache)
        except Exception as exc:
            print(f"[thor_patch] CodePredictor fast hook failed: {exc}", file=sys.stderr, flush=True)

    skip_cfg = os.environ.get("SOPHON_THOR_FLOW_SKIP_ZERO_CFG", "0") == "1"
    flow_bf16 = os.environ.get("SOPHON_THOR_FLOW_BF16", "0") == "1"
    flow_cg = os.environ.get("SOPHON_THOR_FLOW_CG", "0") == "1"
    if skip_cfg or flow_bf16 or flow_cg:
        try:
            _install_flow_fast(skip_cfg, flow_bf16, flow_cg)
        except Exception as exc:
            print(f"[thor_patch] Flow fast hook failed: {exc}", file=sys.stderr, flush=True)


install()
