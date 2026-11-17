"""Sophon-Realtime-4B-Flash runtime wrapper around vLLM-Omni with /health and /metrics."""
import asyncio
import base64
import io
import json
import logging
import os
import signal
import struct
import subprocess
import time
from contextlib import asynccontextmanager

import httpx
import soundfile as sf
import websockets
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from token_metrics import TokenMetrics

LOG = logging.getLogger("sophon_runtime")
TOKEN_METRICS = TokenMetrics("input", "output", "requests", "work")

PORT = int(os.environ.get("AIPOD_METRICS_PORT", "18009"))
VLLM_PORT = int(os.environ.get("SOPHON_VLLM_PORT", str(PORT + 1)))
MODEL_DIR = os.environ.get("SOPHON_MODEL", "/model")
MODEL_ID = os.environ.get("SOPHON_MODEL_ID", "Sophon-Realtime-4B-Flash")
DEPLOY_CONFIG = os.environ.get("SOPHON_DEPLOY_CONFIG", "/app/config/sophon_realtime_flash_4b.yaml")
ORIN_DEPLOY_CONFIG = "/app/config/sophon_realtime_flash_4b_orin.yaml"


def _wav_duration_from_b64(b64_str: str) -> float:
    """Return WAV duration in seconds from a base64 chunk in O(1) via RIFF header."""
    if os.environ.get("SOPHON_O1_WAV", "1") == "1" and len(b64_str) >= 60:
        hdr = base64.b64decode(b64_str[:60])
        if len(hdr) >= 44 and hdr[:4] == b"RIFF" and hdr[8:12] == b"WAVE":
            byte_rate = struct.unpack_from("<I", hdr, 28)[0]
            data_size = struct.unpack_from("<I", hdr, 40)[0]
            if byte_rate > 0 and data_size > 0:
                return data_size / float(byte_rate)
    w, sr = sf.read(io.BytesIO(base64.b64decode(b64_str)), dtype="float32")
    return len(w) / float(sr) if sr else 0.0


def _is_orin_gpu() -> bool:
    """Return True when running on Jetson AGX Orin (sm_87) or SOPHON_DEVICE=orin."""
    if os.environ.get("SOPHON_DEVICE", "").lower() == "orin":
        return True
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (8, 7)
    except Exception:
        return False


def _resolve_deploy_config() -> str:
    explicit = os.environ.get("SOPHON_DEPLOY_CONFIG")
    if explicit:
        return explicit
    if _is_orin_gpu() and os.path.isfile(ORIN_DEPLOY_CONFIG):
        return ORIN_DEPLOY_CONFIG
    return DEPLOY_CONFIG


def _build_vllm_cmd():
    cmd = [
        "vllm", "serve", MODEL_DIR,
        "--omni",
        "--port", str(VLLM_PORT),
        "--host", "127.0.0.1",
        "--trust-remote-code",
        "--init-timeout", "3000",
        "--stage-init-timeout", "3000",
        "--deploy-config", _resolve_deploy_config(),
    ]
    if os.environ.get("SOPHON_NSYS_PROFILE") == "1":
        nsys_bin = os.environ.get("SOPHON_NSYS_BIN", "/opt/nsight/target-linux-sbsa-armv8/nsys")
        nsys_out = os.environ.get("SOPHON_NSYS_OUTPUT", "/cache/nsys_current")
        cmd = [
            nsys_bin, "profile",
            "--start-later=true",
            "--session-new=sophon",
            "--kill=none",
            "--trace=cuda,nvtx,osrt",
            "--trace-fork-before-exec=true",
            "--cuda-graph-trace=node",
            "--force-overwrite=true",
            "-o", nsys_out,
        ] + cmd
    return cmd


@asynccontextmanager
async def lifespan(app: FastAPI):
    env = os.environ.copy()
    env.setdefault("SOPHON_MODEL", MODEL_DIR)
    env.setdefault("SOPHON_PORT", str(VLLM_PORT))
    env.setdefault("SOPHON_BRIDGE_PORT", str(VLLM_PORT + 700))
    env.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    if os.environ.get("SOPHON_MPS", "0") == "1" and os.path.exists("/usr/bin/nvidia-cuda-mps-control"):
        pipe_dir = "/tmp/nvidia-mps/pipe"
        log_dir = "/tmp/nvidia-mps/log"
        os.makedirs(pipe_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        env["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir
        env["CUDA_MPS_LOG_DIRECTORY"] = log_dir
        subprocess.run(["/usr/bin/nvidia-cuda-mps-control", "-d"], env=env, check=False)
    if _is_orin_gpu():
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        env.pop("NVIDIA_TF32_OVERRIDE", None)
        env["OMP_NUM_THREADS"] = os.environ.get("ORIN_OMP_NUM_THREADS", "2")
        env.setdefault("MALLOC_ARENA_MAX", "2")
        env.setdefault("VLLM_ENGINE_ITERATION_TIMEOUT_S", "3600")
        cu13_lib = "/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib"
        ld_path = env.get("LD_LIBRARY_PATH", "")
        if cu13_lib not in ld_path.split(":"):
            env["LD_LIBRARY_PATH"] = f"{cu13_lib}:{ld_path}" if ld_path else cu13_lib

    app.state.vllm_proc = subprocess.Popen(_build_vllm_cmd(), env=env)
    app.state.client = httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{VLLM_PORT}",
        timeout=httpx.Timeout(300.0, connect=5.0),
    )
    try:
        yield
    finally:
        await app.state.client.aclose()
        proc = app.state.vllm_proc
        if proc and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()


app = FastAPI(title="Sophon-Realtime-4B-Flash Runtime", lifespan=lifespan, docs_url=None, redoc_url=None)


@app.get("/health")
async def health():
    proc = app.state.vllm_proc
    if proc is not None and proc.poll() is not None:
        raise HTTPException(503, f"vLLM process exited with code {proc.returncode}")
    try:
        r = await app.state.client.get("/health", timeout=3.0)
        if r.status_code != 200:
            raise HTTPException(503, "vLLM health not ready")
        m = await app.state.client.get("/v1/models", timeout=3.0)
        if m.status_code != 200 or not m.json().get("data"):
            raise HTTPException(503, "Model not loaded yet")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, f"vLLM unreachable: {exc}") from exc
    return {"status": "ready", "model": MODEL_ID}


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(TOKEN_METRICS.render(), media_type="text/plain; version=0.0.4")


@app.get("/v1/models")
async def list_models():
    r = await app.state.client.get("/v1/models")
    return JSONResponse(r.json(), status_code=r.status_code)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.body()
    try:
        payload = json.loads(body)
    except Exception:
        payload = {}
    stream = bool(payload.get("stream", False))

    if not stream:
        r = await app.state.client.post("/v1/chat/completions", content=body, headers={"Content-Type": "application/json"})
        if r.status_code == 200:
            try:
                data = r.json()
                usage = data.get("usage") or {}
                TOKEN_METRICS.add(
                    requests=1,
                    input=int(usage.get("prompt_tokens", 0)),
                    output=int(usage.get("completion_tokens", 0)),
                )
            except Exception:
                TOKEN_METRICS.add(requests=1)
        return JSONResponse(r.json(), status_code=r.status_code)

    async def event_stream():
        audio_sec = 0.0
        prompt_tok = 0
        comp_tok = 0
        gen_chunks = 0
        completed = False
        async with app.state.client.stream("POST", "/v1/chat/completions", content=body, headers={"Content-Type": "application/json"}) as r:
            if r.status_code != 200:
                err = await r.aread()
                yield err
                return
            async for line in r.aiter_lines():
                yield (line + "\n").encode("utf-8")
                if line.startswith("data:"):
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        completed = True
                        continue
                    try:
                        chunk = json.loads(raw)
                        if chunk.get("usage"):
                            prompt_tok = max(prompt_tok, int(chunk["usage"].get("prompt_tokens", 0)))
                            comp_tok = max(comp_tok, int(chunk["usage"].get("completion_tokens", 0)))
                        modality = chunk.get("modality")
                        if modality == "audio":
                            for ch in chunk.get("choices", []):
                                c = (ch.get("delta") or {}).get("content")
                                if c:
                                    gen_chunks += 1
                                    audio_sec += _wav_duration_from_b64(c)
                        elif modality == "text":
                            for ch in chunk.get("choices", []):
                                if (ch.get("delta") or {}).get("content"):
                                    gen_chunks += 1
                    except Exception:
                        pass
        if completed:
            TOKEN_METRICS.add(requests=1, input=prompt_tok, output=comp_tok or gen_chunks, work=round(audio_sec, 4))

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.websocket("/v1/realtime")
async def realtime_ws(ws: WebSocket):
    await ws.accept()
    TOKEN_METRICS.add(requests=1)
    target_url = f"ws://127.0.0.1:{VLLM_PORT}/v1/realtime"
    if ws.url.query:
        target_url += "?" + ws.url.query
    try:
        async with websockets.connect(target_url, max_size=32 * 1024 * 1024) as upstream:
            async def c2u():
                try:
                    while True:
                        msg = await ws.receive()
                        if msg["type"] == "websocket.disconnect":
                            break
                        if msg.get("text") is not None:
                            await upstream.send(msg["text"])
                        elif msg.get("bytes") is not None:
                            await upstream.send(msg["bytes"])
                except WebSocketDisconnect:
                    pass

            async def u2c():
                try:
                    async for msg in upstream:
                        if isinstance(msg, bytes):
                            await ws.send_bytes(msg)
                        else:
                            await ws.send_text(msg)
                except Exception:
                    pass

            await asyncio.gather(c2u(), u2c(), return_when=asyncio.FIRST_COMPLETED)
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass
