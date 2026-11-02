"""End-to-end: the official OpenAI Python client against a RUNNING voice-server, real models.

Skipped unless VOICE_SERVER_E2E_TOKEN is set:
  export VOICE_SERVER_E2E_TOKEN="$(./voicectl tokens add e2e-test | tail -1)"
  .venv/bin/python -m pytest tests/e2e -v
  ./voicectl tokens revoke e2e-test
VOICE_SERVER_URL defaults to http://127.0.0.1:8100/v1.
"""

from __future__ import annotations

import io
import os
import subprocess
import time
import wave
from pathlib import Path

import pytest

TOKEN = os.environ.get("VOICE_SERVER_E2E_TOKEN")
URL = os.environ.get("VOICE_SERVER_URL", "http://127.0.0.1:8100/v1")
pytestmark = pytest.mark.skipif(not TOKEN, reason="set VOICE_SERVER_E2E_TOKEN to run against a live server")


@pytest.fixture(scope="module")
def client():
    from openai import OpenAI
    return OpenAI(api_key=TOKEN, base_url=URL, timeout=120, max_retries=0)


def _words(s: str) -> set[str]:
    return {w.strip(".,!?'\"").lower() for w in s.split()}


def test_wrong_token_is_refused():
    from openai import AuthenticationError, OpenAI
    with pytest.raises(AuthenticationError):
        OpenAI(api_key="vs-wrong", base_url=URL, max_retries=0).models.list()


def test_models_and_aliases(client):
    ids = {m.id for m in client.models.list().data}
    assert {"whisper-1", "tts-1", "tts-1-hd"} <= ids


@pytest.mark.parametrize("lang,text", [
    ("en", "The weather is lovely today, let us go for a walk in the park."),
    ("fr", "Il fait très beau aujourd'hui, allons nous promener dans le parc."),
])
@pytest.mark.parametrize("tts_model", ["tts-1", "tts-1-hd"])
def test_round_trip_speech_then_transcription(client, lang, text, tts_model):
    # Speech as Ogg/Opus (what Slack voice notes are), then transcribe it back with whisper-1.
    t = time.perf_counter()
    speech = client.audio.speech.create(model=tts_model, voice="alloy", input=text, response_format="opus")
    tts_s = time.perf_counter() - t
    data = speech.read()
    assert data[:4] == b"OggS" and len(data) > 2000
    t = time.perf_counter()
    out = client.audio.transcriptions.create(model="whisper-1", file=("note.ogg", io.BytesIO(data)),
                                             language=lang, response_format="json")
    stt_s = time.perf_counter() - t
    overlap = len(_words(out.text) & _words(text)) / len(_words(text))
    print(f"{tts_model} {lang}: tts {tts_s:.2f}s stt {stt_s:.2f}s overlap {overlap:.0%}: {out.text}")
    assert overlap >= 0.7


def test_text_format_like_hermes(client):
    # Hermes sends response_format="text" for whisper-1 and gets a plain string back.
    wav = client.audio.speech.create(model="tts-1", voice="nova", input="Testing one two three.",
                                     response_format="wav").read()
    with wave.open(io.BytesIO(wav)) as w:
        assert w.getnchannels() == 1 and w.getnframes() > 0
    text = client.audio.transcriptions.create(model="whisper-1", file=("a.wav", io.BytesIO(wav)),
                                              response_format="text")
    assert isinstance(text, str) and ("three" in text.lower() or "3" in text)


def test_streaming_pcm_like_hermes_voice_mode(client):
    # Hermes' streaming TTS: response_format=pcm, rate from the X-Audio-Sample-Rate header.
    t = time.perf_counter()
    first_at, total = None, 0
    with client.audio.speech.with_streaming_response.create(
            model="tts-1-hd", voice="alloy", response_format="pcm",
            input="Here is the first sentence of a longer reply. And here is a second sentence after it.") as r:
        rate = int(r.headers["x-audio-sample-rate"])
        for chunk in r.iter_bytes():
            if first_at is None and chunk:
                first_at = time.perf_counter() - t
            total += len(chunk)
    assert rate > 0 and total / 2 / rate > 2.0
    print(f"pcm stream: first audio after {first_at:.2f}s, {total / 2 / rate:.1f}s of audio at {rate} Hz")


def test_voice_note_file_from_bench_data(client):
    clip = Path(__file__).resolve().parents[2] / "bench" / "data" / "clips" / "fr_voicenote.ogg"
    if not clip.exists():
        pytest.skip("run bench/fetch_data.py for the reference clips")
    t = time.perf_counter()
    out = client.audio.transcriptions.create(model="whisper-1", file=clip.open("rb"), language="fr")
    print(f"10 s French voice note: {time.perf_counter() - t:.2f}s: {out.text}")
    assert len(out.text.split()) > 10


def test_ffmpeg_present():
    assert subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0
