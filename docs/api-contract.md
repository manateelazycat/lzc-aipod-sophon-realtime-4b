# Sophon-Realtime-4B-Flash Runtime API Contract

## Service Identity
- **Package ID**: `cloud.lazycat.aipod.sophon-realtime-4b`
- **Model ID**: `Sophon-Realtime-4B-Flash`
- **Inference Engine**: `vLLM` (`0.30.0`) + `vLLM-Omni` (`0.30.0`) + `duplexjev-vllm` (`0.3.0`) + `sophon-realtime-flash` (`1.4.0`)
- **Port**: `18009` (Wrapper API & Metrics), internal vLLM-Omni on `18010`

## Health & Metrics
- `GET /health`:
  - Returns HTTP `200` with `{"status": "ready", "model": "Sophon-Realtime-4B-Flash"}` **only** when the vLLM-Omni subprocess is alive, `/health` returns 200, and `/v1/models` lists the loaded model.
  - Returns HTTP `503` during startup or if the model/process is not ready.
- `GET /metrics`:
  - Returns HTTP `200`, `Content-Type: text/plain; version=0.0.4`.
  - Exposes cumulative Prometheus counters:
    - `aipod:prompt_tokens_total`
    - `aipod:generation_tokens_total`
    - `aipod:requests_total`
    - `aipod:work_total` (cumulative generated audio duration in seconds)

## Inference Endpoints
- `GET /v1/models`: Lists loaded model metadata.
- `POST /v1/chat/completions`:
  - OpenAI-compatible multimodal chat completions (`modalities: ["text", "audio"]`, `stream: true|false`).
  - Supports audio input (`input_audio` base64 WAV) and streaming SSE text + base64 WAV audio chunks (24 kHz).
- `WebSocket /v1/realtime`:
  - OpenAI Realtime API compatible full-duplex audio WebSocket endpoint.

## Timeouts & Limits
- Health check start period: `300s`.
- Request timeout: `300s`.
- Recommended concurrency: up to `8` concurrent sessions on AGX Thor / AGX Orin.
