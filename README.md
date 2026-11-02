# tts-stt-voice-server

> **Vibe-coded by [Gérard Métrailler](https://www.orionplaybook.com) with Claude Opus 5.5 (Anthropic)** · [www.orionplaybook.com](https://www.orionplaybook.com)

> **⚠️ Disclaimer:** this software is provided **as is**, without warranty of any kind: no guarantee that it is useful, correct or fit for any purpose, and **no support** is offered. Benchmark figures are indicative only. Use it at your own risk. See [LICENSE](LICENSE).

A local, OpenAI-compatible **speech-to-text and text-to-speech** server for Apple Silicon Macs.
It runs the models natively on the GPU with [MLX](https://github.com/ml-explore/mlx) (through
[mlx-audio](https://github.com/Blaizzy/mlx-audio)), so tools running in Docker containers, which
cannot use the Mac's GPU, get fast local voice over the network, the same way they would use a
local LLM server such as LM Studio.

- `POST /v1/audio/transcriptions` and `POST /v1/audio/speech`, shaped exactly like OpenAI's, so
  any client with an "OpenAI" speech provider works unchanged (`whisper-1`, `tts-1`, `tts-1-hd`,
  `alloy`, `nova`... are all accepted).
- Accepts what chat apps send: Ogg/Opus voice notes, m4a, mp3, wav, webm, flac (decoded by ffmpeg).
- Speech comes back streamed, sentence by sentence, in mp3, opus, aac, flac, wav or raw pcm.
- One bearer token per consumer, stored hashed. Logs name the consumer, never the token, and
  never the text spoken or transcribed (unless you turn on a debug flag).
- Runs as a launchd user agent, from its own pinned environment inside this folder.

Why each model was chosen, with measurements: [BENCHMARKS.md](BENCHMARKS.md). Why this stack:
[EVALUATION.md](EVALUATION.md).

## Contents

- [Models](#models)
- [Requirements](#requirements)
- [Install](#install)
- [voicectl, the command line](#voicectl-the-command-line)
- [Tokens](#tokens)
- [Operate](#operate)
- [API](#api)
- [Configuration](#configuration)
- [Using with Hermes Agent](#using-with-hermes-agent)
- [Other clients](#other-clients)
- [Updates](#updates)
- [Uninstall](#uninstall)
- [What is in this repository](#what-is-in-this-repository)
- [Troubleshooting](#troubleshooting)
- [Design notes](#design-notes)
- [Development](#development)
- [Licences](#licences)

## Models

| Role | Name in requests | Model | Licence | Languages | Memory (resident) | Speed (M5 Pro) |
|---|---|---|---|---|---|---|
| Speech to text | `whisper-1` (default) | [Whisper large-v3 turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo-asr-6bit), 6-bit | MIT | 99, auto-detected | 1.7 GB | 10 s voice note in 0.3 s |
| Text to speech | `tts-1`, `tts-1-hd` (default) | [Kokoro-82M](https://huggingface.co/mlx-community/Kokoro-82M-bf16) | Apache-2.0 | English, French | 1.0 GB | first audio in 0.2 s |
| Text to speech, other languages | `chatterbox`, and automatically for languages Kokoro lacks (German) | [Chatterbox multilingual](https://huggingface.co/mlx-community/chatterbox-8bit) (Resemble AI), 8-bit | MIT | 23 incl. English, French, German | 1.9 GB, only while in use | first audio in 1.1–1.2 s (3–4 s on first use) |

Kokoro won a blind listening test against Chatterbox (full and 8-bit), Chatterbox Turbo,
Qwen3-TTS and VoxCPM2 (see [BENCHMARKS.md](BENCHMARKS.md)); it is also the fastest and smallest.
Chatterbox is not loaded at start: it loads on its first request and unloads after 10 idle
minutes. The service at rest: about 1.7 GB, peaking around 4 GB while it works.

Everything here is configuration: which model `tts-1` and `tts-1-hd` mean, which voice OpenAI's
voice names map to, which model takes over for which language, what is preloaded. Any of
Kokoro's 54 voices can also be asked for by name (`af_bella`, `am_michael`, `bf_emma`...);
`GET /v1/audio/voices` lists them.
Also configured, on demand: **Qwen3-ASR 1.7B** (Apache-2.0), more accurate than Whisper turbo,
especially on long recordings (2.8 GB while loaded; ask for it with `model=qwen3-asr`, e.g.
`voicectl transcribe meeting.m4a --model qwen3-asr`). It loads on first use and unloads after
10 idle minutes.

All of them are downloaded from Hugging Face on install (into the standard Hugging Face cache,
`~/.cache/huggingface`) and loaded at start. After that the service runs offline
(`HF_HUB_OFFLINE=1` in the launchd agent).

## Requirements

- A Mac with Apple Silicon (M1 or later), macOS 14 or later. 16 GB of memory is the practical
  minimum; see [BENCHMARKS.md](BENCHMARKS.md) for what each model uses.
- `ffmpeg` on the `PATH` (for example `brew install ffmpeg`).
- About 5 GB of disk: 2 GB of models, the rest for the Python environment and uv's cache.
- `/usr/bin/python3` (ships with the Xcode command line tools). It is used once, to install
  [uv](https://github.com/astral-sh/uv) into this folder; uv then installs its own Python 3.12
  here too. Nothing is installed system-wide.

## Install

```bash
git clone https://github.com/orionplaybook/tts-stt-voice-server.git
cd tts-stt-voice-server
scripts/install.sh
```

`install.sh` creates `.tools/` (uv), `.python/` and `.venv/` inside the folder from the pinned
`uv.lock`, downloads the configured models at the commits pinned in `models.lock.json`, writes
`~/Library/LaunchAgents/local.voice-server.plist` and starts it. The first start takes a minute
or two while the models load and warm up; `install.sh` waits until `/health` answers.

Then create a token for each program that will use the service (see [Tokens](#tokens)), and check
it all works: `./voicectl selftest`.

To set everything up without starting it: `scripts/install.sh --no-start`.

## voicectl, the command line

Everything you can do with the service is one `voicectl` command, in the project folder. Put it on
your `PATH` once to use it from anywhere:

```bash
mkdir -p ~/.local/bin && ln -sf "$PWD/voicectl" ~/.local/bin/voicectl
```

| Command | What it does |
|---|---|
| `voicectl status` | Is the service running, which models are loaded |
| `voicectl say "Bonjour !"` | Speak text through the speakers. `-o reply.mp3` saves it instead (format from the extension: mp3, ogg, wav, flac, aac); `-f file.txt` or a pipe reads the text; `--voice`, `--model`, `--language`, `--speed` |
| `voicectl transcribe note.ogg [more files]` | Print the transcript. `--language fr`, `--prompt "names to expect"`, `--format text\|json\|verbose_json\|srt\|vtt`, `-o out.srt` (a folder for several files) |
| `voicectl voices [--all]` | Voices per TTS model, which OpenAI names map to them, and the model's own voice names |
| `voicectl models` | Configured models: loaded or not, pinned version, size, every name they answer to, and downloads no longer used |
| `voicectl models discover` | Recent MLX speech models on Hugging Face (`--kind stt\|tts`, `--days`, `--search`), with licence |
| `voicectl models add NAME REPO` / `remove NAME` | Add a model to your config (downloaded and pinned; on demand unless `--preload`) / take it out again and delete its files |
| `voicectl models delete REPO` / `clean` | Delete one unused download / every download voice-server no longer uses |
| `voicectl bench stt [REPO...]` / `tts REPO` / `memory` / `report` | Benchmark models (accuracy, speed, memory; listening samples for TTS) |
| `voicectl config` | The effective configuration: files, address, limits, models, fallbacks |
| `voicectl tokens add NAME` / `list` / `revoke NAME` | Consumer tokens (see [Tokens](#tokens)) |
| `voicectl selftest` | Live check: health, 401 without a token, and speak-then-transcribe in every language |
| `voicectl check` | Health and auth only, with an exit code for monitoring |
| `voicectl start` / `stop` / `restart` / `logs` | Operate the launchd agent; `restart` after a config change |
| `voicectl update [--check]` | Update packages and models, tested, rolled back on failure (see [Updates](#updates)) |
| `voicectl autoupdate enable` / `disable` / `status` | The same, weekly at night |
| `voicectl install` / `upgrade` / `uninstall [--purge]` | The scripts in `scripts/` |

`voicectl --help` and `voicectl <command> --help` describe every option. Examples:

```bash
voicectl say "Your meeting starts in ten minutes."
voicectl say "Bonjour ! Le rapport est prêt." --voice ff_siwis -o bonjour.mp3
echo "Ein kurzer Test." | voicectl say -                      # German goes to Chatterbox
voicectl transcribe ~/Downloads/voice-note.ogg --language fr
voicectl transcribe interview.m4a --format srt -o interview.srt
voicectl say "Testing a different voice." --voice am_michael
```

Talking to the service needs a token. voicectl uses `--token`, else `$VOICE_SERVER_TOKEN`, else
its own: on first use it creates a token for the consumer `voicectl` and keeps it next to the
token store, readable by you only (`voicectl tokens revoke voicectl` removes it; a new one is
made on next use). To use another machine's service: `--url http://host:8100` or
`$VOICE_SERVER_URL`.

### Trying and adding new models

New MLX speech models appear often. The loop is:

```bash
voicectl models discover --kind tts --days 30          # what is new, with licences
voicectl bench tts mlx-community/NEW-MODEL --voice NAME --languages en,fr
                                                       # speed, memory, intelligibility, and
                                                       # listening samples under a new letter in samples/
voicectl bench stt mlx-community/NEW-ASR-MODEL         # word error rate EN/FR/DE, speed, memory
voicectl models add newvoice mlx-community/NEW-MODEL --languages en,fr --voice NAME
voicectl restart
voicectl say "Hello, testing the new voice." --model newvoice
```

To make it what clients get by default, point the alias in `config.toml` at it (for example
`"tts-1-hd" = "newvoice"` under `[tts.aliases]`), then `voicectl restart`. To undo:
`voicectl models remove newvoice`. After benchmarking, `voicectl models clean` deletes every
downloaded model that is not in use.

Some things to know:

- A model only works if mlx-audio supports its family. `voicectl bench` tells you quickly: an
  unsupported model fails to load. `voicectl update` brings newer mlx-audio releases.
- `models add` writes to `config.local.toml` (your machine's file) between marker comments, so
  `models remove` can take it out cleanly. Models defined in `config.toml` are edited by hand.
- Check the licence `discover` and `add` show: several new models are non-commercial.
- Benchmarks refuse to start with less than 14 GB of free memory (`--min-free-gb`, `--force`) and
  abort any model that grows past 12 GB, so they cannot push the Mac into swap.
- `models clean` and `delete` only touch models downloaded by voice-server or its benchmarks,
  never other tools' entries in the shared Hugging Face cache.

## Tokens

Every request except `GET /health` needs `Authorization: Bearer <token>`. Create one token per
consumer, so each can be revoked on its own:

```bash
voicectl tokens add my-agent       # prints the token once: store it in that program
voicectl tokens list               # names and creation dates, never the tokens
voicectl tokens revoke my-agent    # stops working immediately
```

Tokens are stored as SHA-256 hashes in `~/.config/voice-server/tokens.json` (mode 600; change the
location with `tokens_file` in the config). The running service notices changes to the file;
no restart needed. A missing or wrong token gets `401`.

## Operate

```bash
voicectl status       # launchd state and /health
voicectl restart      # after a config change
voicectl stop | start
voicectl logs         # follow the log
voicectl check        # health 200 + 401 without a token; exit code for cron/monitoring
voicectl selftest     # also speaks and transcribes in every language
```

(The same is available without voicectl as `scripts/voice-server.sh` and `scripts/health-check.sh`.)

- Logs: `~/Library/Logs/voice-server/voice-server.log`, rotated at 10 MB, 5 files kept. One line
  per request: status, endpoint, consumer, duration, model, audio length, language, character
  count. `launchd.log` next to it only receives crash output.
- launchd restarts the service if it exits (at most every 30 s) and starts it at login.
- `GET /health` (no token) returns the status, the models loaded right now, and a stable `version`
  string (plus `versions`, its details: server, mlx-audio, mlx, each model's and helper repo's pinned commit) that
  changes only when an update is applied. Monitoring can diff `version`; `models` changes as
  on-demand models load and unload. Nothing else, no secrets.

## API

| Endpoint | Body | Returns |
|---|---|---|
| `GET /health` | | `{"status":"ok","models":[...],"version":"...","versions":{...}}`, no token needed |
| `GET /v1/models` | | configured models, aliases, voices |
| `GET /v1/audio/voices` | optional `?model=` | configured voices per model, the OpenAI names mapped to them, and the model's own voice names |
| `POST /v1/audio/transcriptions` | multipart: `file` (≤ 25 MB), `model`, `language`, `prompt`, `response_format` (`json`, `text`, `verbose_json`, `srt`, `vtt`), `temperature` | transcript |
| `POST /v1/audio/speech` | JSON: `model`, `input` (≤ 4096 characters), `voice`, `response_format` (`mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`), `speed` (0.25–4), `instructions`, `stream_format` (`audio`, `sse`), `language` | audio, streamed |
| `WS /v1/realtime?intent=transcription` | OpenAI Realtime transcription events (`session.update`, `input_audio_buffer.append`/`commit`/`clear`) | `...input_audio_transcription.completed` per commit |

Details that matter for clients:

- **Model names.** `whisper-1` (and `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`,
  `gpt-transcribe`, `gpt-live-transcribe`) mean the default STT model. `tts-1`, `tts-1-hd` and
  `gpt-4o-mini-tts` mean the TTS models set in `[tts.aliases]` (Kokoro by default). Configured names and the Hugging Face
  repo ids work too. An empty `model` means the default.
- **Voices.** Every OpenAI voice name (`alloy`, `nova`, `shimmer`, `onyx`...) maps to a
  configured voice; native voice names of the model (for example Kokoro's `af_bella`) are passed
  through.
- **Language of speech.** If the request names none (`language`, or `lang_code` as some clients
  send it), the server detects English, French or German from the text, per request. A language
  the model does not speak goes to its `fallback` model (Kokoro → Chatterbox for German).
- **Streaming.** Long input is split into sentences; the first sentence's audio is sent while
  the rest is generated. `pcm` is 24 kHz 16-bit mono like OpenAI's, and every response carries
  `X-Audio-Sample-Rate`. `wav` and `flac` are sent complete (their headers need the length).
- **Realtime.** A subset: manual commits only (`turn_detection: null`); each commit is
  transcribed as a whole and returned as one delta plus `completed`. No partial words while
  speaking.
- **Errors** use OpenAI's shape `{"error": {"message", "type", "code"}}`: `400` bad input,
  `401` token, `413` upload too large, `429` too many requests queued for a model (`Retry-After`),
  `503` waited too long for the GPU, `504` inference timeout.

```bash
TOKEN=vs-...
curl -s http://127.0.0.1:8100/v1/audio/transcriptions -H "Authorization: Bearer $TOKEN" \
  -F file=@voice-note.ogg -F model=whisper-1 -F response_format=text
curl -s http://127.0.0.1:8100/v1/audio/speech -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"model":"tts-1-hd","voice":"alloy","input":"Bonjour ! Comment ça va ?"}' -o reply.mp3
```

## Configuration

`config.toml` in this folder; an optional `config.local.toml` next to it is merged on top (keep
machine-specific values there, such as `tokens_file`). Restart after a change
(`scripts/voice-server.sh restart`); `.venv/bin/voice-server --check-config` validates first.

```toml
[server]
host = "127.0.0.1"          # see "Network" below
port = 8100
tokens_file = "~/.config/voice-server/tokens.json"
log_content = false         # true logs transcripts and speech input: debugging only
max_upload_mb = 25
max_input_chars = 4096
queue_timeout_s = 30        # waited longer for the GPU -> 503
request_timeout_s = 180
mlx_cache_limit_gb = 0.25   # GPU memory MLX keeps for reuse while a request runs
release_after_request = true # hand it back to macOS after every request

[stt]
default = "whisper-turbo"
[stt.aliases]
"whisper-1" = "whisper-turbo"

[stt.models.whisper-turbo]
repo = "mlx-community/whisper-large-v3-turbo-asr-6bit"
preload = true              # load at start and keep warm
idle_unload_s = 0           # > 0: unload after this many idle seconds, reload on demand
max_queue = 8               # requests waiting or running for this model; beyond -> 429

[tts]
default = "kokoro"
[tts.aliases]
"tts-1" = "kokoro"
"tts-1-hd" = "kokoro"            # point it at "chatterbox" to make that the HD voice
[tts.voice_aliases]
alloy = "default"                # OpenAI voice name -> configured voice

[tts.models.kokoro]
repo = "mlx-community/Kokoro-82M-bf16"
languages = ["en", "fr"]
fallback = "chatterbox"          # any other language goes to this model
default_voice = "default"
[tts.models.kokoro.voices.default]
en = { speaker = "af_heart" }    # one speaker per language
fr = { speaker = "ff_siwis" }
[tts.models.kokoro.voices.calm]  # add as many named voices as you like
en = { speaker = "bf_emma" }
fr = { speaker = "ff_siwis" }

[tts.models.chatterbox]
repo = "mlx-community/chatterbox-8bit"
languages = ["en", "fr", "de"]
preload = false                  # load on first use
idle_unload_s = 600              # and unload after 10 idle minutes
```

A voice can also clone a reference clip on models that support it:
`"*" = { ref_audio = "voices/my-voice.wav", ref_text = "What the clip says." }` (path relative to
the config file). Only use voices you have the right to use.

### Network

By default the service listens on `127.0.0.1` only. That is enough for programs on the same Mac
**and** for Docker Desktop containers: Docker Desktop forwards `host.docker.internal` to the
Mac's loopback. Do not expose it to the internet.

To reach it from other machines, tokens and audio must not travel in clear text, so the server
refuses a non-loopback `host` over plain HTTP. Either:

- **serve HTTPS directly:** set `host = "0.0.0.0"`, `tls_cert` and `tls_key` (PEM files, e.g. from
  your own CA or `mkcert`); clients then use `https://<mac-name>:8100/v1`. voicectl and the
  scripts on the Mac follow automatically; or
- **keep `127.0.0.1` and put a TLS reverse proxy in front** (Caddy, nginx, a NAS proxy), which
  terminates HTTPS and forwards to `http://127.0.0.1:8100`; or
- set `allow_plain_http_remote = true` if you knowingly accept plain HTTP on a trusted network.

macOS may ask once to allow incoming connections. Tokens are required in every mode.

### Limits

Uploads up to 25 MB (`max_upload_mb`), audio up to 30 minutes once decoded (`max_audio_minutes`;
a tiny, highly compressed file cannot expand to hours of audio in memory), speech input up to
4,096 characters, realtime PCM at a standard rate (8, 16, 22.05, 24, 32, 44.1 or 48 kHz).

## Using with Hermes Agent

[Hermes Agent](https://github.com/NousResearch/hermes-agent) (Nous Research) has "openai" STT
and TTS providers that work with this server unchanged. If Hermes runs in Docker, give it a
stable name for the server, for example a tiny relay container on Hermes' network:

```yaml
services:
  voice:
    image: alpine/socat:latest
    command: tcp-listen:8100,fork,reuseaddr tcp-connect:host.docker.internal:8100
    extra_hosts: ["host.docker.internal:host-gateway"]
    read_only: true
    cap_drop: [ALL]
```

Then in Hermes' `config.yaml`:

```yaml
stt:
  provider: openai
  language: ""                 # auto-detect; Hermes' default "en" forces English
  openai:
    model: whisper-1
    base_url: http://voice:8100/v1
tts:
  provider: openai
  openai:
    model: tts-1-hd            # whatever [tts.aliases] maps it to (Kokoro by default)
    voice: alloy
    base_url: http://voice:8100/v1
voice:
  client_direct: false         # keep the desktop app from calling the server directly
```

and in Hermes' environment (`.env`):

```
VOICE_TOOLS_OPENAI_KEY=<token>               # used for both directions
STT_OPENAI_BASE_URL=http://voice:8100/v1     # required: see below
```

Things worth knowing:

- With the key in the environment, Hermes ignores `stt.openai.base_url` and sends speech-to-text
  to `STT_OPENAI_BASE_URL`, or to api.openai.com if that is unset, with your token. Set
  `STT_OPENAI_BASE_URL`. (`tts.openai.base_url` is honoured.)
- Hermes Desktop voice chat speaks replies only when **Read replies aloud** is checked in the
  microphone button's menu. It is a local Desktop setting; `voice.auto_tts` in the config only
  seeds it the first time a Desktop connects.

- Point Hermes at a plain host name (like `voice` above), not `host.docker.internal`: Hermes
  treats names ending in `.internal`, `.local` or `.lan` as keyless local servers and sends a
  placeholder key, which this server rejects.
- Hermes' voice mode streams speech sentence by sentence with `response_format=pcm`; the server
  answers in 24 kHz and says so in `X-Audio-Sample-Rate`.
- `stt.streaming: true` makes Hermes use `/v1/realtime`; it works, but this server transcribes
  each committed utterance as a whole, so it adds little over the default batch path.
- The wake word is detected by Hermes itself; it never reaches this server.

## Other clients

Anything that speaks the OpenAI audio API: set the base URL to `http://<host>:8100/v1` and the API
key to a token. With the OpenAI Python SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8100/v1", api_key="vs-...")
text = client.audio.transcriptions.create(model="whisper-1", file=open("note.ogg", "rb")).text
client.audio.speech.create(model="tts-1-hd", voice="alloy", input=text).write_to_file("reply.mp3")
```

n8n's OpenAI node, Open WebUI and others take the same two settings.

## Updates

What the service runs on is pinned exactly: Python packages in `uv.lock` (every package, direct
and indirect), models in `models.lock.json` (one Hugging Face commit per model, including helper
repos such as Chatterbox's speech tokenizer). Nothing changes until an update runs, and the
running service never downloads anything (it starts with `HF_HUB_OFFLINE=1`).

```bash
voicectl update --check      # what would change; changes nothing
voicectl update              # do it
```

An update:

1. backs up `pyproject.toml`, `uv.lock` and `models.lock.json`;
2. moves each package to its newest release **only if that release is at least 7 days old**
   (`--cooldown-days`; younger releases wait for a later run, nothing ever moves backwards), and
   each model to its newest commit at least that old. The waiting period lets a broken or
   tampered release be noticed and pulled upstream before it reaches you;
3. installs the new versions, runs the unit tests, restarts the service and runs
   `voicectl selftest` (a real speak-then-transcribe round trip in every language);
4. if any step fails, restores the backups, reinstalls the previous versions, restarts, and checks
   the service again.

Every run writes `~/Library/Logs/voice-server/update-status.json` (result: `updated`,
`up-to-date`, `rolled-back` or `failed`, with the changes, any error, and the tool versions) and
appends to `update.log` there. A monitoring job can read the JSON.

### Automatic weekly updates

Off by default. To turn it on (a launchd job, `local.voice-server.update`):

```bash
voicectl autoupdate enable                              # Sundays at 02:30
voicectl autoupdate enable --weekday wed --time 03:15 --cooldown-days 14
voicectl autoupdate status                              # schedule and last result
voicectl autoupdate disable
```

The service is down for about 15 seconds while it restarts. Updated `uv.lock` /
`models.lock.json` are left for you to commit (the status says when they are uncommitted).

Not covered by updates: ffmpeg (yours to update, e.g. `brew upgrade ffmpeg`), uv itself (pinned in
`scripts/common.sh`), and the choice of models (that is configuration: edit `config.toml`, then
`scripts/upgrade.sh`, which fetches new models before restarting).

### By hand

To pin a specific version yourself: edit the floor in `pyproject.toml` or run
`.tools/bin/uv lock --upgrade-package NAME`, review the `uv.lock` diff, then `scripts/upgrade.sh`
(applies the lock files, fetches models, restarts, checks health). After a `git pull`, run
`scripts/upgrade.sh` the same way. To roll back by hand, check out the previous `uv.lock` and
`models.lock.json` and run it again.

## Uninstall

```bash
scripts/uninstall.sh           # stop and remove the launchd agent; keeps tokens, models, logs
scripts/uninstall.sh --purge   # also removes .venv/.tools/.python, logs, the token file and
                               # the configured models from the Hugging Face cache
```

## What is in this repository

Only this project's own files. No third-party code and no models: Python packages are installed
by `scripts/install.sh` from `uv.lock` (into `.venv/`, ignored by git), models are downloaded into
the Hugging Face cache from `models.lock.json`, and the test audio for the benchmarks is fetched by
`bench/fetch_data.py`. No tokens either: they live hashed outside the project (`tokens_file`).

`scripts/check-repo.sh` checks what git would commit and refuses tokens, keys, token stores,
virtualenvs and installed packages, wheels and compiled libraries, model weights, and any file over
5 MB. Run it before every commit.

## Troubleshooting

| Symptom | Check |
|---|---|
| `install.sh`: "ffmpeg not found" | Install ffmpeg (`brew install ffmpeg`) and re-run. |
| `/health` does not answer after start | `scripts/voice-server.sh logs` and `~/Library/Logs/voice-server/launchd.log`. First start loads and warms every preloaded model: allow a couple of minutes. |
| Hermes voice notes fail; Hermes' gateway log shows `401 Incorrect API key provided`, and nothing reaches `voicectl logs` | The request went to api.openai.com, not here: with the key in the environment Hermes ignores `stt.openai.base_url`. Set `STT_OPENAI_BASE_URL=http://voice:8100/v1` (see [Using with Hermes Agent](#using-with-hermes-agent)), restart Hermes, and replace the token, which was sent to OpenAI. |
| Spoken replies stop after the first sentence when using speakers | The client's barge-in hears the reply through the speakers and interrupts it (the log shows the first chunk served and no further request). Use headphones, or mute the mic between turns. Not a server problem. |
| Hermes Desktop voice chat transcribes but never speaks | Turn on **Read replies aloud** in the menu on Desktop's microphone button (see [Using with Hermes Agent](#using-with-hermes-agent)). |
| Every request gets `401` | `voicectl tokens list` (is there a token?), and check the client sends `Authorization: Bearer <token>`. Hermes: see the `.internal` note above. |
| `429` | More than `max_queue` requests waiting for that model; raise it or slow the client. |
| `503 waited more than 30 s` | The GPU is busy (long speech, or an LLM hogging it). Raise `queue_timeout_s`. |
| French voice note transcribed as English | The client forces `language=en` (Hermes' default). Send `language=fr`, or none for auto-detect. |
| Wrong voice language | Pass `language` (or `lang_code`) in the speech request; detection only knows EN/FR/DE. |
| Container cannot connect | From the container: `curl http://host.docker.internal:8100/health`. If that fails, see [Network](#network). |
| "refusing to listen on ... over plain HTTP" at start | A non-loopback `host` needs `tls_cert`/`tls_key` or a proxy; see [Network](#network). |
| `400 audio longer than 30 minutes` | Raise `max_audio_minutes`, or split the recording. |
| Memory pressure | `idle_unload_s` on rarely used models, `preload = false`, or drop a TTS tier; see BENCHMARKS.md. |
| An automatic update did not happen or was rolled back | `voicectl autoupdate status` (last result and error), then `~/Library/Logs/voice-server/update.log`. |
| After a macOS update the agent does not start | `scripts/install.sh --no-start && scripts/voice-server.sh start` rewrites the plist. |

## Design notes

Choices that are not obvious from the code:

- **mlx-audio as a library, not its bundled server.** Its server has no authentication and lets
  any caller download and load any model. See [EVALUATION.md](EVALUATION.md).
- **One inference thread.** MLX inference is serialized per process. Each model has an admission
  limit and a queue timeout instead of an unbounded backlog. Streamed speech is generated one
  chunk per job, so a transcription can run between two sentences of a long reply.
- **Sentence splitting before synthesis**, so time to first audio does not grow with the length of
  the reply; models that stream natively also stream within the sentence.
- **Whisper on long audio** is fed independent ~10 s windows cut at pauses: Whisper turbo's own
  long-form decoding skipped or repeated whole sentences in testing.
- **Small memory footprint by default**: quantized Whisper (6-bit) and Chatterbox (8-bit), which
  measured as accurate as their full-precision versions at about half the memory; Chatterbox only
  loaded while it is needed; and an MLX
  buffer cache capped at 0.25 GB and handed back to macOS after every request. Without the cap a
  0.6B model measured 7–8 GB, and one candidate passed 50 GB.
- **Tokens hashed with plain SHA-256**: they are 256-bit random values, so a slow password hash
  adds nothing.
- **ffmpeg for all audio**: one well-known decoder for every container chat apps produce, run as
  a subprocess with pipes (no temporary audio files).
- **Language detection for speech** is a small function-word heuristic for EN/FR/DE, enough to
  choose the voice and pronunciation, with no extra dependency.

## Development

```bash
.tools/bin/uv sync --all-groups                     # dev and benchmark dependencies too
.venv/bin/python -m pytest                          # unit tests: fake models, no GPU needed
export VOICE_SERVER_E2E_TOKEN="$(./voicectl tokens add e2e-test | tail -1)"
.venv/bin/python -m pytest tests/e2e -v -s           # OpenAI client against the running service
./voicectl tokens revoke e2e-test
```

Benchmarks: `voicectl bench ...` (see [Trying and adding new models](#trying-and-adding-new-models)),
which runs the scripts in `bench/` (`fetch_data.py` downloads the public FLEURS test clips on first
use); results in `bench/results/`, summary in [BENCHMARKS.md](BENCHMARKS.md), listening samples in
[samples/](samples/).

## Licences

The code in this repository: MIT, see [LICENSE](LICENSE).

Models are downloaded at install time and keep their own licences (all listed in
[Models](#models) allow commercial use). Libraries: mlx and mlx-audio (MIT), FastAPI (MIT),
uvicorn (BSD). Kokoro's French pronunciation uses espeak-ng (GPL-3.0), installed as the
`espeakng-loader` wheel and loaded as a separate library; it is not part of this repository.
Benchmark audio comes from Google's FLEURS dataset (CC-BY-4.0) and is downloaded by
`bench/fetch_data.py`, not stored here.
