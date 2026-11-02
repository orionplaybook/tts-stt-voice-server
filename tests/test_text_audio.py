import numpy as np
import pytest

from voiceserver import audio
from voiceserver.engines import normalize_language
from voiceserver.text import detect_language, split_sentences


def test_split_sentences_merges_short_and_keeps_abbreviations():
    s = split_sentences("Hi. Dr. Smith arrived at 9.30 today. Is that right? Yes! Mme. Dupont aussi.")
    assert s == ["Hi. Dr. Smith arrived at 9.30 today.", "Is that right? Yes! Mme. Dupont aussi."]


def test_split_sentences_cuts_runaways():
    long = "word, " * 200
    parts = split_sentences(long, max_chars=100)
    assert all(len(p) <= 101 for p in parts) and "".join(parts).count("word") == 200


@pytest.mark.parametrize("text,lang", [
    ("Your meeting starts at nine and the report is ready.", "en"),
    ("Ta réunion commence à neuf heures et le rapport est prêt.", "fr"),
    ("Dein Meeting beginnt um neun und der Bericht ist fertig.", "de"),
    ("OK", "en"),
])
def test_detect_language(text, lang):
    assert detect_language(text) == lang


@pytest.mark.parametrize("text,allowed,lang", [
    ("Hola, ¿cómo estás? El informe está listo para la reunión.", ["es"], "es"),
    ("Hola, ¿cómo estás? El informe está listo para la reunión.", ["en", "es"], "es"),
    ("Your report is ready for the meeting.", ["en", "es"], "en"),
    ("Il rapporto è pronto per la riunione di domani.", ["en", "fr", "it"], "it"),
    ("O relatório está pronto, e a reunião é amanhã.", ["en", "pt"], "pt"),
    ("Het rapport is klaar voor de vergadering.", ["en", "nl"], "nl"),
    ("こんにちは", ["ja"], "ja"),            # a language the detector does not know: the model's own
    ("12345", ["fr", "en"], "fr"),            # no clue at all: the model's first language
])
def test_detect_language_other_model_languages(text, allowed, lang):
    assert detect_language(text, allowed) == lang


def test_detect_language_respects_allowed():
    assert detect_language("Dein Meeting beginnt um neun und der Bericht ist fertig.", ["en", "fr"]) in ("en", "fr")


@pytest.mark.parametrize("raw,code", [("fr", "fr"), ("FR", "fr"), ("fr-CA", "fr"), ("French", "fr"),
                                      ("auto", None), (None, None), ("en_US", "en")])
def test_normalize_language(raw, code):
    assert normalize_language(raw) == code


def test_decode_ogg_opus_to_16k_float(ogg_bytes):
    x = audio.decode(ogg_bytes)
    assert x.dtype == np.float32 and abs(x.size - 32000) < 400 and 0.05 < np.abs(x).max() <= 1.0


def test_decode_rejects_garbage():
    with pytest.raises(audio.AudioError):
        audio.decode(b"definitely not audio")


def test_stream_encoder_mp3():
    enc = audio.StreamEncoder("mp3", 24000)
    out = enc.feed(np.zeros(24000, dtype=np.float32)) + enc.close()
    assert len(out) > 1000


def test_pcm16_resamples():
    out = audio.pcm16(np.zeros(44100, dtype=np.float32), 44100)
    assert len(out) == 24000 * 2


def test_decode_stops_at_max_seconds(wav_bytes):
    assert audio.decode(wav_bytes, max_seconds=5).size == 32000
    with pytest.raises(audio.AudioError, match="longer than"):
        audio.decode(wav_bytes, max_seconds=1)


def test_refuses_plain_http_off_loopback(cfg, monkeypatch):
    import voiceserver.__main__ as main
    ran = []
    monkeypatch.setattr(main.uvicorn, "run", lambda *a, **k: ran.append(k))
    monkeypatch.setattr(main, "create_app", lambda cfg: None)
    with pytest.raises(SystemExit, match="refusing to listen on 0.0.0.0 over plain HTTP"):
        main.main(["--config", str(cfg.source[0]), "--host", "0.0.0.0"])
    assert not ran
    main.main(["--config", str(cfg.source[0])])  # loopback: fine
    assert ran[-1]["host"] == "127.0.0.1" and ran[-1]["ssl_certfile"] is None


def test_tls_when_configured(cfg, monkeypatch, tmp_path):
    import voiceserver.__main__ as main
    ran = []
    monkeypatch.setattr(main.uvicorn, "run", lambda *a, **k: ran.append(k))
    monkeypatch.setattr(main, "create_app", lambda cfg: None)
    text = cfg.source[0].read_text().replace("[server]", f'[server]\nhost = "0.0.0.0"\ntls_cert = "{tmp_path}/c.pem"\ntls_key = "{tmp_path}/k.pem"')
    cfg.source[0].write_text(text)
    main.main(["--config", str(cfg.source[0])])
    assert ran[-1]["ssl_certfile"] == f"{tmp_path}/c.pem" and ran[-1]["host"] == "0.0.0.0"
