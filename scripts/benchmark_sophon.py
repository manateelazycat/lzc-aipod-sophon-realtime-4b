"""Reproducible benchmark for Sophon-Realtime streaming audio+text inference."""
import argparse
import base64
import concurrent.futures
import io
import json
import statistics
import time
import urllib.request
import array
import wave


def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def run_one(url, model, sys_prompt, wav_b64, lang, max_tokens=200):
    body = {
        "model": model,
        "stream": True,
        "modalities": ["text", "audio"],
        "language": lang,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "<|audio|>"},
                {"type": "input_audio", "input_audio": {"data": wav_b64, "format": "wav"}},
            ]},
        ],
    }
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    ttfa = None
    text_parts = []
    pcm_parts = []
    sr = 24000
    chunk_times = []
    with urllib.request.urlopen(req, timeout=180) as r:
        for line in r:
            s = line.decode("utf-8", errors="replace").strip()
            if not s.startswith("data:") or s[5:].strip() == "[DONE]":
                continue
            d = json.loads(s[5:])
            for ch in d.get("choices", []):
                c = (ch.get("delta") or {}).get("content")
                if not c:
                    continue
                now = time.perf_counter()
                if d.get("modality") == "audio":
                    raw_wav = base64.b64decode(c)
                    with wave.open(io.BytesIO(raw_wav), "rb") as wf:
                        sr = wf.getframerate()
                        nframes = wf.getnframes()
                        sw = wf.getsampwidth()
                        frames = wf.readframes(nframes)
                    if nframes > 0:
                        if ttfa is None:
                            ttfa = now - t0
                        chunk_times.append(now - t0)
                        if sw == 2:
                            arr = array.array("h", frames)
                            m = max((abs(x) for x in arr), default=0) / 32768.0
                        else:
                            m = 0.1
                        pcm_parts.append((nframes, m))
                else:
                    if ttft is None:
                        ttft = now - t0
                    text_parts.append(c)
    e2e = time.perf_counter() - t0
    total_frames = sum(x[0] for x in pcm_parts)
    audio_dur = total_frames / float(sr) if sr else 0.0
    peak = max((x[1] for x in pcm_parts), default=0.0)
    text = "".join(text_parts)
    ok = bool(text.strip()) and audio_dur > 0.3 and peak > 0.005 and ttfa is not None
    return {
        "ok": ok,
        "ttft_s": ttft,
        "ttfa_s": ttfa,
        "e2e_s": e2e,
        "audio_dur_s": audio_dur,
        "rtf": (e2e / audio_dur) if audio_dur > 0 else None,
        "audio_chunks": len(pcm_parts),
        "text_chars": len(text),
        "peak_amp": peak,
        "text_preview": text[:120],
    }


def summarize(runs):
    ok_runs = [r for r in runs if r["ok"]]
    if not ok_runs:
        return {"ok_count": 0, "total": len(runs)}
    ttfa = [r["ttfa_s"] for r in ok_runs]
    ttft = [r["ttft_s"] for r in ok_runs if r["ttft_s"] is not None]
    e2e = [r["e2e_s"] for r in ok_runs]
    rtf = [r["rtf"] for r in ok_runs if r["rtf"] is not None]
    aud = [r["audio_dur_s"] for r in ok_runs]
    return {
        "ok_count": len(ok_runs),
        "total": len(runs),
        "ttfa_mean_s": round(statistics.mean(ttfa), 4),
        "ttfa_p50_s": round(percentile(ttfa, 50), 4),
        "ttfa_p95_s": round(percentile(ttfa, 95), 4),
        "ttft_mean_s": round(statistics.mean(ttft), 4) if ttft else None,
        "e2e_mean_s": round(statistics.mean(e2e), 4),
        "e2e_p50_s": round(percentile(e2e, 50), 4),
        "e2e_p95_s": round(percentile(e2e, 95), 4),
        "audio_dur_mean_s": round(statistics.mean(aud), 4),
        "rtf_mean": round(statistics.mean(rtf), 4) if rtf else None,
        "rtf_p95": round(percentile(rtf, 95), 4) if rtf else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--sys-prompt-json", required=True)
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sys_prompts = json.load(open(args.sys_prompt_json, encoding="utf-8"))
    sys_prompt = sys_prompts[args.lang]
    wav_b64 = base64.b64encode(open(args.wav, "rb").read()).decode()
    model = json.load(urllib.request.urlopen(args.url + "/v1/models", timeout=10))["data"][0]["id"]

    for _ in range(args.warmup):
        run_one(args.url, model, sys_prompt, wav_b64, args.lang)

    single_runs = []
    for _ in range(args.runs):
        single_runs.append(run_one(args.url, model, sys_prompt, wav_b64, args.lang))

    conc_t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(run_one, args.url, model, sys_prompt, wav_b64, args.lang) for _ in range(args.concurrency * 2)]
        conc_runs = [f.result() for f in futs]
    conc_wall = time.perf_counter() - conc_t0
    total_audio = sum(r["audio_dur_s"] for r in conc_runs if r["ok"])

    report = {
        "model": model,
        "lang": args.lang,
        "warmup": args.warmup,
        "single_runs": single_runs,
        "single_summary": summarize(single_runs),
        "concurrency": args.concurrency,
        "concurrent_runs": conc_runs,
        "concurrent_summary": summarize(conc_runs),
        "concurrent_wall_s": round(conc_wall, 4),
        "concurrent_audio_throughput_x": round(total_audio / conc_wall, 4) if conc_wall > 0 else 0.0,
    }
    out_txt = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_txt)
    print(json.dumps({
        "single_summary": report["single_summary"],
        "concurrent_summary": report["concurrent_summary"],
        "concurrent_audio_throughput_x": report["concurrent_audio_throughput_x"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
