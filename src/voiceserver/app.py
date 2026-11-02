"""OpenAI-compatible HTTP API.

  GET  /health                      no auth: {"status": "ok", "models": [loaded model names]}
  GET  /v1/models                   configured models and their aliases
  POST /v1/audio/transcriptions     multipart upload -> json | text | verbose_json | srt | vtt
  POST /v1/audio/speech             JSON -> mp3 | opus | aac | flac | wav | pcm, streamed
  WS   /v1/realtime                 OpenAI Realtime transcription subset (manual commit)

Every route except /health needs "Authorization: Bearer <token>" from the token store.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Receive, Scope, Send

from . import __version__, audio, engines
from .config import Config, ModelConfig
from .runtime import Busy, QueueTimeout, Runtime
from .text import detect_language
from .tokens import TokenStore

log = logging.getLogger("voiceserver")

STT_FORMATS = {"json", "text", "verbose_json", "srt", "vtt"}
REALTIME_RATES = {8000, 16000, 22050, 24000, 32000, 44100, 48000}  # PCM rates accepted on /v1/realtime


def _error(status: int, message: str, code: str | None = None, etype: str = "invalid_request_error",
           headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": etype, "param": None, "code": code}},
                        status_code=status, headers=headers)


class BodyLimit:
    """Reject request bodies over a limit while they stream in (Content-Length can be absent)."""

    def __init__(self, app: ASGIApp, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared and declared.isdigit() and int(declared) > self.limit:
            return await _error(413, f"request body larger than {self.limit // 2**20} MB")(scope, receive, send)
        seen = 0

        async def counted() -> dict:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.limit:
                    raise _TooLarge()
            return message

        try:
            await self.app(scope, counted, send)
        except _TooLarge:
            await _error(413, f"request body larger than {self.limit // 2**20} MB")(scope, receive, send)


class _TooLarge(Exception):
    pass


class SpeechRequest(BaseModel):
    model: str = ""
    input: str
    voice: str = ""
    response_format: str = "mp3"
    speed: float = Field(1.0, ge=0.25, le=4.0)
    instructions: str | None = None
    stream_format: str = "audio"            # "audio" or "sse" (OpenAI gpt-4o-mini-tts)
    language: str | None = None             # extension: force a language
    lang_code: str | None = None            # extension: what Hermes sends from tts.openai.language

    model_config = {"extra": "ignore"}


def create_app(cfg: Config, runtime: Runtime | None = None, tokens: TokenStore | None = None) -> FastAPI:
    runtime = runtime or Runtime(cfg)
    tokens = tokens or TokenStore(cfg.server.tokens_file)
    max_upload = int(cfg.server.max_upload_mb * 2**20)
    max_audio_s = cfg.server.max_audio_minutes * 60

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if warning := tokens.check_permissions():
            log.warning(warning)
        if tokens.count() == 0:
            log.warning("no tokens in %s: every request will get 401 (add one with voice-tokens add)",
                        tokens.path)
        await runtime.start()
        log.info("voice-server %s ready on %s:%s", __version__, cfg.server.host, cfg.server.port)
        yield
        await runtime.stop()

    app = FastAPI(title="voice-server", version=__version__, lifespan=lifespan, docs_url=None,
                  redoc_url=None, openapi_url=None)
    # multipart overhead on top of the audio itself
    app.add_middleware(BodyLimit, limit=max_upload + 2**20)
    app.state.runtime = runtime

    # --- plumbing ---------------------------------------------------------------------------

    def consumer_of(auth: str | None) -> str | None:
        if not auth or not auth.lower().startswith("bearer "):
            return None
        return tokens.consumer(auth[7:].strip())

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        started = time.perf_counter()
        request.state.consumer = None
        request.state.info = {}
        if request.url.path != "/health":
            consumer = consumer_of(request.headers.get("authorization"))
            if consumer is None:
                log.info("401 %s %s from %s", request.method, request.url.path,
                         request.client.host if request.client else "?")
                return _error(401, "missing or invalid API token", "invalid_api_key",
                              headers={"WWW-Authenticate": "Bearer"})
            request.state.consumer = consumer
        response = await call_next(request)
        if request.url.path != "/health":
            extra = " ".join(f"{k}={v}" for k, v in request.state.info.items())
            log.info("%s %s %s consumer=%s %.2fs %s", response.status_code, request.method,
                     request.url.path, request.state.consumer, time.perf_counter() - started, extra)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation(_request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", []) if p not in ("body",))
        return _error(400, f"{where}: {first.get('msg', 'invalid request')}".strip(": "))

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException):
        return _error(exc.status_code, str(exc.detail))

    @app.exception_handler(Busy)
    async def busy(_request: Request, exc: Busy):
        return _error(429, str(exc), "rate_limit_exceeded", "rate_limit_error", {"Retry-After": "2"})

    @app.exception_handler(QueueTimeout)
    async def queue_timeout(_request: Request, exc: QueueTimeout):
        return _error(503, str(exc), "overloaded", "server_error", {"Retry-After": "5"})

    @app.exception_handler(asyncio.TimeoutError)
    async def timeout(_request: Request, _exc: asyncio.TimeoutError):
        return _error(504, "inference timed out", "timeout", "server_error")

    def model_or_400(name: str | None, kind: str) -> ModelConfig:
        m = cfg.resolve(name, kind)
        if m is None:
            known = sorted([n for n, mc in cfg.models.items() if mc.kind == kind] +
                           [a for a, t in cfg.aliases.items() if cfg.models[t].kind == kind])
            raise HTTPException(400, f"unknown {kind.upper()} model {name!r}; available: {', '.join(known)}")
        return m

    # --- routes ---------------------------------------------------------------------------

    versions = _versions(cfg)
    version = _version_string(versions)

    @app.get("/health")
    async def health():
        # "models" is what is loaded right now (on-demand models come and go); "version" and
        # "versions" are fixed at start and change only when an update is applied, so a nightly
        # check can diff "version" without seeing on-demand loads as changes.
        return {"status": "ok", "models": runtime.loaded(), "version": version, "versions": versions}

    @app.get("/v1/models")
    async def list_models():
        data = []
        for m in cfg.models.values():
            entry: dict[str, Any] = {
                "id": m.name, "object": "model", "created": 0, "owned_by": "voice-server",
                "kind": m.kind, "repo": m.repo, "loaded": m.name in runtime.loaded(),
                "aliases": sorted(a for a, t in cfg.aliases.items() if t == m.name),
                "default": m.name in (cfg.stt_default, cfg.tts_default)}
            if m.kind == "tts":
                entry["voices"] = sorted(m.voices)
                entry["default_voice"] = m.default_voice
                entry["languages"] = m.languages
                entry["streaming"] = engines.supports_streaming(m.repo)
            data.append(entry)
        for alias, target in cfg.aliases.items():
            data.append({"id": alias, "object": "model", "created": 0, "owned_by": "voice-server",
                         "kind": cfg.models[target].kind, "alias_of": target})
        return {"object": "list", "data": data, "voice_aliases": cfg.voice_aliases}

    @app.get("/v1/audio/voices")
    async def voices(model: str | None = None):
        """Configured voices per TTS model (and the speaker each uses per language), the OpenAI
        names that map to them, and the model's own voice names, any of which can be requested."""
        out = []
        for m in cfg.models.values():
            if m.kind != "tts" or (model and cfg.resolve(model, "tts") is not m):
                continue
            out.append({
                "model": m.name, "aliases": sorted(a for a, t in cfg.aliases.items() if t == m.name),
                "languages": m.languages, "fallback": m.fallback or None, "default_voice": m.default_voice,
                "voices": {v: {lang: spec.speaker or ("reference clip" if spec.ref_audio else "built-in default")
                               for lang, spec in table.items()} for v, table in m.voices.items()},
                "openai_voice_names": sorted(a for a, t in cfg.voice_aliases.items() if t in m.voices),
                "native_voices": engines.native_voices(m.repo)})
        return {"object": "list", "data": out}

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        request: Request,
        file: UploadFile = File(...),
        model: str = Form(""),
        language: str | None = Form(None),
        prompt: str | None = Form(None),
        response_format: str = Form("json"),
        temperature: float | None = Form(None),
    ):
        response_format = (response_format or "json").lower()
        if response_format not in STT_FORMATS:
            raise HTTPException(400, f"response_format must be one of {', '.join(sorted(STT_FORMATS))}")
        m = model_or_400(model, "stt")
        data = await file.read()
        await file.close()
        if len(data) > max_upload:
            raise HTTPException(413, f"file larger than {cfg.server.max_upload_mb:g} MB")
        if not data:
            raise HTTPException(400, "file is empty")
        try:
            pcm = await asyncio.to_thread(audio.decode, data, max_seconds=max_audio_s)
        except audio.AudioError as e:
            raise HTTPException(400, str(e)) from None
        del data
        want_segments = response_format in ("verbose_json", "srt", "vtt")
        result = await runtime.run(m, lambda mdl: engines.transcribe(
            mdl, m.repo, pcm, language=language, prompt=prompt, temperature=temperature,
            want_segments=want_segments))
        request.state.info.update(model=m.name, audio_s=f"{result.duration:.1f}",
                                  lang=result.language or "?", chars=len(result.text))
        if cfg.server.log_content:
            log.debug("transcript consumer=%s: %s", request.state.consumer, result.text)
        if response_format == "text":
            return PlainTextResponse(result.text + "\n")
        if response_format == "json":
            return {"text": result.text}
        if response_format == "verbose_json":
            return {"task": "transcribe", "language": engines.LANG_NAMES.get(result.language or "", result.language or "").lower(),
                    "duration": round(result.duration, 2), "text": result.text, "segments": result.segments}
        return PlainTextResponse(_subtitles(result.segments, response_format),
                                 media_type="text/vtt" if response_format == "vtt" else "text/plain")

    @app.post("/v1/audio/speech")
    async def speech(request: Request, body: SpeechRequest):
        fmt = body.response_format.lower()
        if fmt not in audio.SPEECH_FORMATS:
            raise HTTPException(400, f"response_format must be one of {', '.join(audio.SPEECH_FORMATS)}")
        text = body.input.strip()
        if not text:
            raise HTTPException(400, "input is empty")
        if len(text) > cfg.server.max_input_chars:
            raise HTTPException(400, f"input longer than {cfg.server.max_input_chars} characters")
        m = model_or_400(body.model, "tts")
        speakable = list(m.languages)
        if speakable and m.fallback:
            fb_langs = cfg.models[m.fallback].languages
            speakable = speakable + [x for x in fb_langs if x not in speakable] if fb_langs else []
        lang = engines.normalize_language(body.language or body.lang_code) or detect_language(text, speakable or None)
        if m.languages and lang not in m.languages:
            fb = cfg.models.get(m.fallback) if m.fallback else None
            if fb is None or (fb.languages and lang not in fb.languages):
                raise HTTPException(400, f"model {m.name!r} does not speak {lang!r} (it speaks {', '.join(m.languages)})")
            m = fb  # e.g. German text sent to an English/French-only voice
        spec = runtime.voice_for(m, body.voice, lang)
        request.state.info.update(model=m.name, voice=body.voice or m.default_voice, lang=lang,
                                  chars=len(text), fmt=fmt)
        if cfg.server.log_content:
            log.debug("speech input consumer=%s: %s", request.state.consumer, text)

        chunks = runtime.stream(m, lambda mdl: engines.synthesize_text(
            mdl, m.repo, text, voice=spec, language=lang, speed=body.speed, instructions=body.instructions))
        # Pull the first chunk before answering, so a failure becomes a proper HTTP error and the
        # response headers can carry the real sample rate.
        try:
            first = await chunks.__anext__()
        except StopAsyncIteration:
            raise HTTPException(500, "the model produced no audio") from None
        except (Busy, QueueTimeout, asyncio.TimeoutError):
            raise
        except Exception as e:  # noqa: BLE001 - surface model errors (unknown voice...) as 400
            await chunks.aclose()
            log.warning("speech failed model=%s: %s", m.name, e)
            raise HTTPException(400, f"speech generation failed: {type(e).__name__}: {e}") from None
        _, sr = first
        out_rate = audio.output_rate(fmt, sr)
        _, media_type = audio.SPEECH_FORMATS[fmt]
        headers = {"X-Audio-Sample-Rate": str(out_rate)}
        if fmt == "pcm":
            media_type = f"audio/pcm; rate={out_rate}"

        if body.stream_format == "sse":
            return StreamingResponse(_sse(first, chunks, sr), media_type="text/event-stream", headers=headers)
        if fmt in ("wav", "flac"):  # need the full length in the header: buffer, then encode once
            parts = [first[0]]
            async for c, _ in chunks:
                parts.append(c)
            data = await asyncio.to_thread(audio.encode, np.concatenate(parts), sr, fmt)
            return Response(data, media_type=media_type, headers=headers)
        return StreamingResponse(_encoded(first, chunks, sr, fmt), media_type=media_type, headers=headers)

    async def _encoded(first, chunks, sr: int, fmt: str):
        if fmt == "pcm":
            yield audio.pcm16(first[0], sr)
            async for c, _ in chunks:
                yield audio.pcm16(c, sr)
            return
        enc = audio.StreamEncoder(fmt, sr)
        try:
            if out := await asyncio.to_thread(enc.feed, first[0]):
                yield out
            async for c, _ in chunks:
                if out := await asyncio.to_thread(enc.feed, c):
                    yield out
            if out := await asyncio.to_thread(enc.close):
                yield out
        finally:
            enc.kill()
            await chunks.aclose()

    async def _sse(first, chunks, sr: int):
        def event(payload: dict) -> str:
            return f"data: {json.dumps(payload)}\n\n"
        yield event({"type": "speech.audio.delta", "audio": base64.b64encode(audio.pcm16(first[0], sr)).decode()})
        async for c, _ in chunks:
            yield event({"type": "speech.audio.delta", "audio": base64.b64encode(audio.pcm16(c, sr)).decode()})
        yield event({"type": "speech.audio.done"})

    # --- realtime transcription -----------------------------------------------------------

    @app.websocket("/v1/realtime")
    async def realtime(ws: WebSocket):
        auth = ws.headers.get("authorization")
        protocols = [p.strip() for p in (ws.headers.get("sec-websocket-protocol") or "").split(",") if p.strip()]
        key_proto = next((p for p in protocols if p.startswith("openai-insecure-api-key.")), None)
        consumer = consumer_of(auth) or (tokens.consumer(key_proto.split(".", 1)[1]) if key_proto else None)
        if consumer is None:
            log.info("401 WS /v1/realtime from %s", ws.client.host if ws.client else "?")
            await ws.close(code=1008, reason="missing or invalid API token")
            return
        await ws.accept(subprotocol="realtime" if "realtime" in protocols else None)
        await _realtime_session(ws, consumer)

    async def _realtime_session(ws: WebSocket, consumer: str):
        started = time.perf_counter()
        session: dict[str, Any] = {"id": "sess_" + uuid.uuid4().hex[:20], "type": "transcription",
                                   "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000},
                                                       "transcription": {"model": cfg.stt_default},
                                                       "turn_detection": None}}}
        buf = bytearray()
        turns = 0

        async def send(payload: dict):
            await ws.send_json({"event_id": "event_" + uuid.uuid4().hex[:16], **payload})

        await send({"type": "session.created", "session": session})
        try:
            while True:
                event = json.loads(await ws.receive_text())
                kind = event.get("type", "")
                if kind in ("session.update", "transcription_session.update"):
                    _merge_session(session, event.get("session") or {})
                    await send({"type": "session.updated", "session": session})
                elif kind == "input_audio_buffer.append":
                    buf.extend(base64.b64decode(event.get("audio") or ""))
                    rate = _rate(session)
                    if len(buf) > min(max_upload, int(max_audio_s * (rate or 24000) * 2)):
                        await send({"type": "error", "error": {
                            "type": "invalid_request_error",
                            "message": f"audio buffer too large (limit {cfg.server.max_audio_minutes:g} minutes)"}})
                        buf.clear()
                elif kind == "input_audio_buffer.clear":
                    buf.clear()
                    await send({"type": "input_audio_buffer.cleared"})
                elif kind == "input_audio_buffer.commit":
                    inp = session["audio"]["input"]
                    rate = _rate(session)
                    if rate is None:
                        buf.clear()
                        await send({"type": "error", "error": {
                            "type": "invalid_request_error", "code": "invalid_value",
                            "message": f"unsupported audio rate; use one of {sorted(REALTIME_RATES)}"}})
                        continue
                    if len(buf) < rate * 2 // 10:  # < 100 ms, as OpenAI does
                        await send({"type": "error", "error": {
                            "type": "invalid_request_error", "code": "input_audio_buffer_commit_empty",
                            "message": "buffer too small: expected at least 100ms of audio"}})
                        continue
                    pcm = np.frombuffer(bytes(buf), dtype="<i2").astype(np.float32) / 32768.0
                    buf.clear()
                    if rate != engines.STT_RATE:
                        n = int(round(pcm.size * engines.STT_RATE / rate))
                        pcm = np.interp(np.linspace(0, pcm.size - 1, n), np.arange(pcm.size), pcm).astype(np.float32)
                    tr = inp.get("transcription") or {}
                    m = cfg.resolve(tr.get("model"), "stt") or cfg.models[cfg.stt_default]
                    langs = tr.get("languages") or []
                    lang = tr.get("language") or (langs[0] if langs else None)
                    item_id = "item_" + uuid.uuid4().hex[:20]
                    await send({"type": "input_audio_buffer.committed", "item_id": item_id, "previous_item_id": None})
                    try:
                        result = await runtime.run(m, lambda mdl: engines.transcribe(
                            mdl, m.repo, pcm, language=lang, prompt=tr.get("prompt")))
                    except (Busy, QueueTimeout, asyncio.TimeoutError) as e:
                        await send({"type": "error", "error": {"type": "server_error", "message": str(e) or "timeout"}})
                        continue
                    turns += 1
                    if result.text:
                        await send({"type": "conversation.item.input_audio_transcription.delta",
                                    "item_id": item_id, "content_index": 0, "delta": result.text})
                    await send({"type": "conversation.item.input_audio_transcription.completed",
                                "item_id": item_id, "content_index": 0, "transcript": result.text})
                    if cfg.server.log_content:
                        log.debug("realtime transcript consumer=%s: %s", consumer, result.text)
                else:
                    await send({"type": "error", "error": {"type": "invalid_request_error",
                                                           "message": f"unsupported event type {kind!r}"}})
        except WebSocketDisconnect:
            pass
        except (json.JSONDecodeError, ValueError) as e:
            await ws.close(code=1003, reason=f"bad message: {e}"[:120])
        finally:
            log.info("WS /v1/realtime consumer=%s turns=%d %.1fs", consumer, turns, time.perf_counter() - started)

    return app


def _rate(session: dict) -> int | None:
    """The session's PCM rate if it is a supported one, else None."""
    try:
        rate = int((session["audio"]["input"].get("format") or {}).get("rate") or 24000)
    except (TypeError, ValueError):
        return None
    return rate if rate in REALTIME_RATES else None


def _versions(cfg: Config) -> dict:
    """Server and engine versions, and the pinned commit of every configured model."""
    from importlib.metadata import PackageNotFoundError, version

    from . import modellock

    def pkg(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return "?"
    pins = modellock.load()

    def pin(repo: str) -> str:
        return (pins.get(repo, {}).get("revision") or "unpinned")[:12]
    # Helper repos a model loads by name (extra_repos, e.g. Chatterbox's speech tokenizer) are
    # updated like models, so they are part of the version too.
    helpers = sorted({r for m in cfg.models.values() for r in m.extra_repos})
    return {"voice-server": __version__, "mlx-audio": pkg("mlx-audio"), "mlx": pkg("mlx"),
            "models": {m.name: pin(m.repo) for m in cfg.models.values()},
            "model_helpers": {r: pin(r) for r in helpers}}


def _version_string(v: dict) -> str:
    """One stable line, e.g. "0.1.0 mlx-audio 0.6.3 mlx 0.32.4 models 3f9a1c2b": the last part is a
    short hash of the model and helper-repo pins, so any model update changes it."""
    import hashlib
    pins = hashlib.sha256(json.dumps([v["models"], v.get("model_helpers", {})], sort_keys=True).encode()).hexdigest()[:8]
    return f"{v['voice-server']} mlx-audio {v['mlx-audio']} mlx {v['mlx']} models {pins}"


def _merge_session(session: dict, update: dict) -> None:
    """Accept both the GA shape (session.audio.input.*) and the beta transcription_session shape."""
    inp = session["audio"]["input"]
    new_inp = ((update.get("audio") or {}).get("input")) or {}
    if "input_audio_format" in update:  # beta: "pcm16"
        inp["format"] = {"type": "audio/pcm", "rate": 24000}
    if "input_audio_transcription" in update:
        new_inp = {**new_inp, "transcription": update["input_audio_transcription"]}
    for key in ("format", "transcription", "turn_detection"):
        if key in new_inp:
            inp[key] = new_inp[key] or ({} if key == "transcription" else new_inp[key])
    if not inp.get("format"):
        inp["format"] = {"type": "audio/pcm", "rate": 24000}


def _ts(seconds: float, sep: str) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _subtitles(segments: list[dict], fmt: str) -> str:
    if fmt == "vtt":
        lines = ["WEBVTT", ""]
        for s in segments:
            lines += [f"{_ts(s['start'], '.')} --> {_ts(s['end'], '.')}", s["text"], ""]
    else:
        lines = []
        for i, s in enumerate(segments, 1):
            lines += [str(i), f"{_ts(s['start'], ',')} --> {_ts(s['end'], ',')}", s["text"], ""]
    return "\n".join(lines)
