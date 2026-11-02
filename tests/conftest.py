"""Test fixtures: a config and fake models, so the HTTP layer is tested without MLX or downloads."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from voiceserver import engines
from voiceserver.app import create_app
from voiceserver.config import load_config
from voiceserver.tokens import TokenStore

CONFIG = """
[server]
tokens_file = "{tokens}"
log_dir = "{tmp}/logs"
max_upload_mb = 1
max_input_chars = 200

[stt]
default = "whisper-turbo"
[stt.aliases]
"whisper-1" = "whisper-turbo"
"gpt-live-transcribe" = "whisper-turbo"
[stt.models.whisper-turbo]
repo = "fake/whisper-large-v3-turbo"
[stt.models.parakeet]
repo = "fake/parakeet-tdt"
preload = false

[tts]
default = "quality"
[tts.aliases]
"tts-1" = "fast"
"tts-1-hd" = "quality"
[tts.voice_aliases]
alloy = "default"
nova = "bright"

[tts.models.fast]
repo = "fake/Kokoro-82M"
languages = ["en", "fr"]
fallback = "quality"
default_voice = "default"
max_queue = 2
[tts.models.fast.voices.default]
en = {{ speaker = "af_heart" }}
fr = {{ speaker = "ff_siwis" }}
[tts.models.fast.voices.bright]
"*" = {{ speaker = "af_bella" }}

[tts.models.quality]
repo = "fake/chatterbox"
languages = ["en", "fr", "de"]
default_voice = "default"
[tts.models.quality.voices.default]
"*" = {{}}
"""


class FakeSTT:
    pass


class FakeTTS:
    sample_rate = 24000


@pytest.fixture
def calls():
    return []


@pytest.fixture
def fake_engines(monkeypatch, calls):
    def transcribe(model, repo, audio, *, language=None, prompt=None, temperature=None, want_segments=False):
        calls.append(("stt", repo, language, prompt, audio.size))
        segs = [{"id": 0, "start": 0.0, "end": 1.5, "text": "hello there"},
                {"id": 1, "start": 1.5, "end": 3.25, "text": "general"}] if want_segments else []
        return engines.Transcript(text="hello there general", language=language or "en",
                                  duration=audio.size / 16000, segments=segs)

    def synthesize(model, repo, text, *, voice, language=None, speed=1.0, instructions=None, stream=False,
                   streaming_interval=0.6):
        calls.append(("tts", repo, text, voice.speaker, language, speed))
        if "explode" in text:
            raise ValueError("unknown voice")
        for _ in range(2):
            yield np.sin(np.linspace(0, 200, 2400)).astype(np.float32) * 0.3, 24000

    monkeypatch.setattr(engines, "load_stt", lambda repo, revision=None: FakeSTT())
    monkeypatch.setattr(engines, "load_tts", lambda repo, revision=None: FakeTTS())
    monkeypatch.setattr(engines, "transcribe", transcribe)
    monkeypatch.setattr(engines, "synthesize", synthesize)
    monkeypatch.setattr(engines, "release", lambda: None)


@pytest.fixture
def cfg(tmp_path: Path):
    tokens = tmp_path / "private" / "tokens.json"
    path = tmp_path / "config.toml"
    path.write_text(textwrap.dedent(CONFIG.format(tokens=tokens, tmp=tmp_path)))
    return load_config(path)


@pytest.fixture
def token(cfg):
    return TokenStore(cfg.server.tokens_file).add("tester")


@pytest.fixture
def client(cfg, token, fake_engines):
    with TestClient(create_app(cfg)) as c:
        c.headers["Authorization"] = f"Bearer {token}"
        yield c


@pytest.fixture
def wav_bytes():
    import subprocess
    out = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                          "sine=frequency=440:duration=2", "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
                         capture_output=True, check=True).stdout
    return out


@pytest.fixture
def ogg_bytes():
    import subprocess
    return subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                           "sine=frequency=300:duration=2", "-ac", "1", "-ar", "48000", "-c:a", "libopus",
                           "-f", "ogg", "pipe:1"], capture_output=True, check=True).stdout
