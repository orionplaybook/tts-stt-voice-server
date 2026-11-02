# Evaluation: which MLX audio stack to build on

Date: 2026-11-02. Time-boxed (under half a day). Hardware: Apple M5 Pro, 64 GB unified memory.

## What the client actually needs (Hermes Agent's "openai" voice providers)

Read from Hermes Agent's source rather than its docs, which leave out paths and formats:

| Use | Request | Notes |
|---|---|---|
| Speech to text (voice mode, dashboard, Slack voice notes) | `POST {base}/audio/transcriptions`, multipart `file`, `model`, `response_format` (`text` for `whisper-1`, else `json`), optional `language`, `prompt` | Batch upload. Ogg/Opus voice notes as-is; retries once as m4a after a 400/5xx. |
| Live transcription (optional, off by default) | WebSocket `{base}/realtime?intent=transcription`, OpenAI Realtime GA transcription session, manual commit | Falls back to the batch path on any failure. |
| Text to speech, whole reply | `POST {base}/audio/speech`, JSON `model`, `voice`, `input`, `response_format` (`mp3`/`opus`), `speed`, `instructions`, `lang_code` | |
| Text to speech, voice mode | Same endpoint, one request per sentence, `response_format=pcm` | Expects 24 kHz int16 unless `X-Audio-Sample-Rate` says otherwise. |
| Wake word | none | On-device (openWakeWord) in Hermes. |

Keys: `VOICE_TOOLS_OPENAI_KEY` (then `OPENAI_API_KEY`) for both directions; base URLs from
`stt.openai.base_url` / `STT_OPENAI_BASE_URL` and `tts.openai.base_url`.

So the service needs the two classic OpenAI audio endpoints done properly (formats, streaming
`pcm`, Ogg/Opus input) and, optionally, a small subset of the Realtime protocol.

## Candidates

### 1. mlx-audio 0.6.3 (Blaizzy, MIT), as a library: **chosen**

- Broadest MLX model coverage by far: 30+ STT families (Whisper, Parakeet, Nemotron ASR, Qwen3-ASR,
  Voxtral, Canary...) and 45+ TTS families (Kokoro, Chatterbox, Qwen3-TTS, VoxCPM2, ZONOS2,
  Higgs, CSM...). Every model on the shortlist below runs through one API.
- Actively maintained (release 2026-10-28, pins `mlx==0.32.4`), MIT, pure Python wheels.
- Model-native streaming for the TTS models that have it (Chatterbox, Qwen3-TTS, ...).

### 1b. mlx-audio's own `mlx_audio.server`: **not used as the service**

It has the right endpoints (`/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/realtime`),
but it is a demo server, not something to put on a network:

- No authentication at all.
- `POST /v1/models` and `DELETE /v1/models` let any caller download and load any Hugging Face
  repo: arbitrary downloads plus an easy way to exhaust memory next to LM Studio.
- Default `response_format` for transcription is mlx-audio's own `ndjson`, a browser UI on
  another port, wide CORS defaults.
- No per-model queue limits, no idle unload, no memory cap.

Wrapping it would mean disabling most of it. Building a thin FastAPI app on mlx-audio's model
API is smaller and every line is ours to review (about 1,400 lines of Python in `src/`, comments included).

### 2. mlx-omni-server 0.5.3

OpenAI-compatible MLX server for LLMs plus audio. Last release May 2026 (six months old), audio
is a side feature built on mlx-whisper and an older TTS set, no auth model of its own. Rejected:
narrower and staler than mlx-audio for this job, and we do not want a second LLM server next to
LM Studio.

### 3. whisper.cpp server (Metal)

Excellent, boring, MIT, fast on Metal. But STT only (a second service would still be needed for
TTS), Whisper only (no Qwen3-ASR/Parakeet to compare), its server speaks its own `/inference`
API with an OpenAI shim that is incomplete, and building it needs CMake/Xcode tooling or
Homebrew. Rejected for this project; it stays the fallback if mlx-audio ever regresses for STT.

### Others looked at briefly

Kokoro-FastAPI (TTS only, one model), various single-model MLX demos. None cover both directions.

## Decision

mlx-audio (pinned) as the inference library, inside a small FastAPI service that adds what a
shared, always-on service needs: bearer tokens per consumer, OpenAI-exact request/response
shapes, ffmpeg decoding and encoding, sentence-level streaming, one serialized inference thread
with per-model queues and timeouts, a capped MLX buffer cache, launchd operation.

Model choices and the measurements behind them: see BENCHMARKS.md.
