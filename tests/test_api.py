"""HTTP API with fake models: formats, aliasing, limits."""

import base64

import numpy as np
import pytest


def test_health_lists_loaded_models_and_stable_versions(client):
    body = client.get("/health", headers={"Authorization": ""}).json()
    assert body["status"] == "ok" and body["models"] == ["fast", "quality", "whisper-turbo"]
    assert set(body) == {"status", "models", "version", "versions"}  # nothing else, no secrets
    v = body["versions"]
    assert set(v) == {"voice-server", "mlx-audio", "mlx", "models", "model_helpers"}
    assert set(v["models"]) == {"whisper-turbo", "parakeet", "fast", "quality"}  # configured, not loaded
    assert body["version"].startswith(v["voice-server"] + " mlx-audio ")


def test_health_version_ignores_on_demand_loading(cfg, token, fake_engines, wav_bytes):
    from fastapi.testclient import TestClient

    from voiceserver.app import create_app
    with TestClient(create_app(cfg)) as c:
        before = c.get("/health").json()
        r = c.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav_bytes)}, data={"model": "parakeet"},
                   headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        after = c.get("/health").json()
    assert "parakeet" not in before["models"] and "parakeet" in after["models"]  # loaded on demand
    assert before["version"] == after["version"] and before["versions"] == after["versions"]


def test_models_lists_aliases(client):
    data = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
    assert data["whisper-1"]["alias_of"] == "whisper-turbo"
    assert data["tts-1"]["alias_of"] == "fast" and data["tts-1-hd"]["alias_of"] == "quality"
    assert data["fast"]["voices"] == ["bright", "default"]
    assert data["parakeet"]["loaded"] is False


@pytest.mark.parametrize("model,repo", [("whisper-1", "fake/whisper-large-v3-turbo"),
                                        ("", "fake/whisper-large-v3-turbo"),
                                        ("parakeet", "fake/parakeet-tdt"),
                                        ("fake/parakeet-tdt", "fake/parakeet-tdt")])
def test_stt_aliases(client, calls, wav_bytes, model, repo):
    r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav_bytes)}, data={"model": model})
    assert r.status_code == 200, r.text
    assert calls[-1][1] == repo


def test_stt_unknown_model(client, wav_bytes):
    r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav_bytes)}, data={"model": "tts-1"})
    assert r.status_code == 400 and "whisper-1" in r.json()["error"]["message"]


def test_stt_ogg_opus_voice_note(client, calls, ogg_bytes):
    r = client.post("/v1/audio/transcriptions", files={"file": ("voice.ogg", ogg_bytes, "audio/ogg")},
                    data={"model": "whisper-1", "language": "fr", "prompt": "Hermes"})
    assert r.status_code == 200 and r.json() == {"text": "hello there general"}
    _, _, language, prompt, samples = calls[-1]
    assert language == "fr" and prompt == "Hermes" and abs(samples - 32000) < 400


@pytest.mark.parametrize("fmt,check", [
    ("text", lambda r: r.text == "hello there general\n"),
    ("json", lambda r: r.json() == {"text": "hello there general"}),
    ("verbose_json", lambda r: r.json()["segments"][1]["end"] == 3.25 and r.json()["duration"] == 2.0),
    ("srt", lambda r: "1\n00:00:00,000 --> 00:00:01,500\nhello there" in r.text),
    ("vtt", lambda r: r.text.startswith("WEBVTT") and "00:00:01.500 --> 00:00:03.250" in r.text),
])
def test_stt_response_formats(client, wav_bytes, fmt, check):
    r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav_bytes)},
                    data={"model": "whisper-1", "response_format": fmt})
    assert r.status_code == 200 and check(r), r.text


def test_stt_bad_format_and_garbage_audio(client, wav_bytes):
    r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav_bytes)},
                    data={"response_format": "xml"})
    assert r.status_code == 400
    r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"not audio at all")})
    assert r.status_code == 400 and "decode" in r.json()["error"]["message"]


def test_stt_upload_cap(client):
    big = b"\0" * (2 * 2**20 + 10)  # limit in the test config is 1 MB
    r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", big)})
    assert r.status_code == 413


@pytest.mark.parametrize("fmt,ctype,magic", [
    ("mp3", "audio/mpeg", (b"ID3", b"\xff\xfb", b"\xff\xf3")),
    ("opus", "audio/ogg", (b"OggS",)),
    ("aac", "audio/aac", (b"\xff\xf1", b"\xff\xf9")),
    ("flac", "audio/flac", (b"fLaC",)),
    ("wav", "audio/wav", (b"RIFF",)),
])
def test_tts_formats(client, fmt, ctype, magic):
    r = client.post("/v1/audio/speech", json={"model": "tts-1", "input": "Hello world, how are you?",
                                              "voice": "alloy", "response_format": fmt})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith(ctype)
    assert r.content.startswith(magic)


def test_tts_pcm_is_24k_int16_with_rate_header(client):
    r = client.post("/v1/audio/speech", json={"input": "Hello world, how are you?", "response_format": "pcm"})
    assert r.status_code == 200
    assert r.headers["x-audio-sample-rate"] == "24000"
    assert "rate=24000" in r.headers["content-type"]
    assert len(r.content) == 2 * 2400 * 2  # two chunks of 2400 samples, 2 bytes each


def test_tts_aliases_and_voices(client, calls):
    client.post("/v1/audio/speech", json={"model": "tts-1", "input": "Hello there, how are you today?",
                                          "voice": "alloy"})
    assert calls[-1][1] == "fake/Kokoro-82M" and calls[-1][3] == "af_heart" and calls[-1][4] == "en"
    client.post("/v1/audio/speech", json={"model": "tts-1", "input": "Bonjour, comment ça va aujourd'hui ?",
                                          "voice": "alloy"})
    assert calls[-1][3] == "ff_siwis" and calls[-1][4] == "fr"
    client.post("/v1/audio/speech", json={"model": "tts-1", "input": "Hello there friend", "voice": "nova"})
    assert calls[-1][3] == "af_bella"
    client.post("/v1/audio/speech", json={"model": "tts-1", "input": "Hello there friend", "voice": "am_adam"})
    assert calls[-1][3] == "am_adam"  # native voice names pass through
    client.post("/v1/audio/speech", json={"model": "tts-1-hd", "input": "Hello there friend"})
    assert calls[-1][1] == "fake/chatterbox"
    client.post("/v1/audio/speech", json={"input": "Hello there friend", "lang_code": "fr", "speed": 1.25})
    assert calls[-1][4] == "fr" and calls[-1][5] == 1.25


def test_tts_streams_sentence_by_sentence(client, calls):
    client.post("/v1/audio/speech", json={"input": "First sentence is here. Second one follows now!"})
    texts = [c[2] for c in calls if c[0] == "tts" and c[2] != "Ready."]
    assert texts == ["First sentence is here.", "Second one follows now!"]


def test_tts_limits_and_errors(client):
    assert client.post("/v1/audio/speech", json={"input": "x" * 201}).status_code == 400
    assert client.post("/v1/audio/speech", json={"input": "  "}).status_code == 400
    assert client.post("/v1/audio/speech", json={"input": "hi", "response_format": "ogg"}).status_code == 400
    assert client.post("/v1/audio/speech", json={"input": "hi", "speed": 9}).status_code == 400
    assert client.post("/v1/audio/speech", json={"input": "hi", "model": "whisper-1"}).status_code == 400
    r = client.post("/v1/audio/speech", json={"model": "tts-1", "input": "Konnichiwa", "language": "ja"})
    assert r.status_code == 400 and "does not speak" in r.json()["error"]["message"]
    r = client.post("/v1/audio/speech", json={"input": "please explode now"})
    assert r.status_code == 400 and "unknown voice" in r.json()["error"]["message"]


def test_tts_sse(client):
    r = client.post("/v1/audio/speech", json={"input": "Hello there, friend.", "stream_format": "sse"})
    events = [line for line in r.text.splitlines() if line.startswith("data: ")]
    assert '"speech.audio.done"' in events[-1] and '"speech.audio.delta"' in events[0]


def test_tts_queue_limit(cfg, token, fake_engines):
    import asyncio

    from voiceserver.runtime import Busy, Runtime

    async def scenario():
        rt = Runtime(cfg)
        m = cfg.models["fast"]
        rt._inflight["fast"] = m.max_queue
        with pytest.raises(Busy):
            await rt.run(m, lambda model: None)
    asyncio.run(scenario())


def test_realtime_transcription_protocol(client, calls):
    pcm = (np.sin(np.linspace(0, 300, 24000)) * 8000).astype("<i2").tobytes()  # 1 s at 24 kHz
    with client.websocket_connect("/v1/realtime?intent=transcription",
                                  headers={"Authorization": client.headers["Authorization"]}) as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {
            "format": {"type": "audio/pcm", "rate": 24000},
            "transcription": {"model": "gpt-live-transcribe", "languages": ["fr"]}, "turn_detection": None}}}})
        assert ws.receive_json()["type"] == "session.updated"
        ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()})
        ws.send_json({"type": "input_audio_buffer.commit"})
        assert ws.receive_json()["type"] == "input_audio_buffer.committed"
        assert ws.receive_json()["type"] == "conversation.item.input_audio_transcription.delta"
        done = ws.receive_json()
        assert done["type"] == "conversation.item.input_audio_transcription.completed"
        assert done["transcript"] == "hello there general"
        ws.send_json({"type": "input_audio_buffer.commit"})  # empty buffer: OpenAI's error
        assert "buffer too small" in ws.receive_json()["error"]["message"]
    stt = [c for c in calls if c[0] == "stt" and c[2] == "fr"]
    assert len(stt) == 1 and abs(stt[0][4] - 16000) < 5  # 1 s resampled from 24 kHz to 16 kHz


def test_tts_language_fallback(client, calls):
    r = client.post("/v1/audio/speech", json={"model": "tts-1", "voice": "alloy",
                                              "input": "Dein Meeting beginnt um neun und der Bericht ist fertig."})
    assert r.status_code == 200
    assert calls[-1][1] == "fake/chatterbox" and calls[-1][4] == "de"
    client.post("/v1/audio/speech", json={"model": "tts-1", "input": "Bonjour, ton rapport est prêt pour demain."})
    assert calls[-1][1] == "fake/Kokoro-82M" and calls[-1][4] == "fr"


def test_voices_endpoint(client):
    data = {m["model"]: m for m in client.get("/v1/audio/voices").json()["data"]}
    assert data["fast"]["voices"]["default"] == {"en": "af_heart", "fr": "ff_siwis"}
    assert data["fast"]["fallback"] == "quality" and "tts-1" in data["fast"]["aliases"]
    assert "alloy" in data["fast"]["openai_voice_names"]
    only = client.get("/v1/audio/voices", params={"model": "tts-1-hd"}).json()["data"]
    assert [m["model"] for m in only] == ["quality"]


def test_bad_fallback_rejected(tmp_path):
    from voiceserver.config import load_config
    from tests.conftest import CONFIG
    text = CONFIG.format(tokens=tmp_path / "t.json", tmp=tmp_path).replace('fallback = "quality"', 'fallback = "nope"')
    path = tmp_path / "config.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match="fallback"):
        load_config(path)


def test_stt_rejects_audio_longer_than_limit(cfg, token, fake_engines, wav_bytes):
    from fastapi.testclient import TestClient

    from voiceserver.app import create_app
    cfg.server.max_audio_minutes = 1 / 60  # 1 second; the clip is 2 s
    with TestClient(create_app(cfg)) as c:
        r = c.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav_bytes)},
                   headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 400 and "longer than" in r.json()["error"]["message"]


def test_realtime_rejects_unsupported_rate(client, calls):
    pcm = b"\x00\x01" * 24000
    with client.websocket_connect("/v1/realtime?intent=transcription",
                                  headers={"Authorization": client.headers["Authorization"]}) as ws:
        ws.receive_json()
        ws.send_json({"type": "session.update", "session": {"audio": {"input": {"format": {"type": "audio/pcm", "rate": 1}}}}})
        ws.receive_json()
        ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()})
        ws.send_json({"type": "input_audio_buffer.commit"})
        err = ws.receive_json()
        assert err["type"] == "error" and "unsupported audio rate" in err["error"]["message"]
    assert not [c for c in calls if c[0] == "stt" and c[4] != 16000]  # nothing was transcribed


def test_health_version_covers_helper_repo_pins(cfg, monkeypatch):
    from voiceserver import app as app_module
    from voiceserver import modellock
    cfg.models["quality"].extra_repos = ["fake/tokenizer"]
    pins = {"fake/chatterbox": {"revision": "a" * 40}, "fake/tokenizer": {"revision": "b" * 40}}
    monkeypatch.setattr(modellock, "load", lambda path=None: dict(pins))
    v1 = app_module._versions(cfg)
    assert v1["model_helpers"] == {"fake/tokenizer": "b" * 12}
    pins["fake/tokenizer"] = {"revision": "c" * 40}  # an update that moves only the helper repo
    v2 = app_module._versions(cfg)
    assert v1["models"] == v2["models"] and app_module._version_string(v1) != app_module._version_string(v2)
