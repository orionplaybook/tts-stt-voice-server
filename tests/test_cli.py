"""voicectl against the in-process app (fake models): commands, token handling, errors."""

import stat

import pytest
from fastapi.testclient import TestClient

from voiceserver import cli
from voiceserver.app import create_app
from voiceserver.tokens import TokenStore


@pytest.fixture(autouse=True)
def isolated_models(monkeypatch, tmp_path):
    """Never touch the real Hugging Face cache, download registry or models.lock.json."""
    from voiceserver import modellock
    state = {"cache": {"fake/old-candidate": 2_000_000_000, "fake/whisper-large-v3-turbo": 1_000_000_000},
             "deleted": [], "synced": 0}
    monkeypatch.setattr(modellock, "REGISTRY", tmp_path / "downloaded.json")
    monkeypatch.setattr(modellock, "LOCK_PATH", tmp_path / "models.lock.json")
    monkeypatch.setattr(modellock, "load", lambda path=None: {})
    monkeypatch.setattr(modellock, "save", lambda repos, path=None: None)
    monkeypatch.setattr(modellock, "cached_repos", lambda: dict(state["cache"]))
    monkeypatch.setattr(modellock, "delete_cached", lambda r: state["deleted"].append(r) or state["cache"].pop(r, 0))
    monkeypatch.setattr(modellock, "sync", lambda cfg, **kw: state.__setitem__("synced", state["synced"] + 1) or [])
    for r in state["cache"]:
        modellock.record_download(r)
    return state


@pytest.fixture
def run(cfg, fake_engines, monkeypatch, capsys):
    """voicectl main() with HTTP routed into a TestClient instead of the network."""
    with TestClient(create_app(cfg)) as tc:
        class InProcess(cli.Client):
            def request(self, method, path, body=None, headers=None):
                h = dict(headers or {})
                if self.token:
                    h["Authorization"] = f"Bearer {self.token}"
                r = tc.request(method, path, content=body, headers=h)
                return r.status_code, {k.lower(): v for k, v in r.headers.items()}, r.content
        monkeypatch.setattr(cli, "Client", InProcess)
        monkeypatch.delenv("VOICE_SERVER_TOKEN", raising=False)
        monkeypatch.delenv("VOICE_SERVER_URL", raising=False)

        def go(*argv):
            code = cli.main([*argv, "--config", str(cfg.source[0])])
            out, err = capsys.readouterr()
            return code, out, err
        yield go


def test_first_use_creates_private_cli_token(run, cfg):
    code, out, err = run("voices")
    assert code == 0 and "fast" in out and "created voicectl's own token" in err
    saved = cfg.server.tokens_file.with_name("voicectl.token")
    assert stat.S_IMODE(saved.stat().st_mode) == 0o600
    assert TokenStore(cfg.server.tokens_file).consumer(saved.read_text().strip()) == "voicectl"
    code, _, err = run("voices")
    assert code == 0 and "created" not in err  # reused, not recreated


def test_say_saves_audio(run, tmp_path):
    target = tmp_path / "hello.ogg"
    code, _, err = run("say", "Hello there, this is a test.", "-o", str(target))
    assert code == 0 and target.read_bytes()[:4] == b"OggS" and "saved" in err


def test_say_reads_file(run, tmp_path, calls):
    src = tmp_path / "reply.txt"
    src.write_text("Bonjour, ceci est un test de la ligne de commande.")
    code, _, _ = run("say", "-f", str(src), "-o", str(tmp_path / "r.mp3"), "--model", "tts-1")
    assert code == 0 and calls[-1][1] == "fake/Kokoro-82M" and calls[-1][4] == "fr"


def test_transcribe_formats(run, tmp_path, wav_bytes, ogg_bytes):
    a, b = tmp_path / "a.wav", tmp_path / "b.ogg"
    a.write_bytes(wav_bytes)
    b.write_bytes(ogg_bytes)
    code, out, _ = run("transcribe", str(a))
    assert code == 0 and out == "hello there general\n"
    code, out, _ = run("transcribe", str(a), str(b), "--format", "srt", "--language", "fr")
    assert code == 0 and out.count("00:00:00,000 --> 00:00:01,500") == 2 and f"== {b}" in out
    outdir = tmp_path / "out"
    code, _, _ = run("transcribe", str(a), str(b), "-o", str(outdir), "--format", "vtt")
    assert (outdir / "a.vtt").read_text().startswith("WEBVTT") and (outdir / "b.vtt").exists()


def test_transcribe_missing_file(run, tmp_path):
    code, _, err = run("transcribe", str(tmp_path / "nope.ogg"))
    assert code == 1 and "no such file" in err


def test_tokens_lifecycle(run, cfg):
    code, out, _ = run("tokens", "add", "n8n")
    token = out.strip()
    assert code == 0 and token.startswith("vs-")
    code, out, _ = run("tokens", "list")
    assert "n8n" in out and token not in out
    code, _, err = run("tokens", "add", "n8n")
    assert code == 1 and "already exists" in err
    assert run("tokens", "revoke", "n8n")[0] == 0
    assert TokenStore(cfg.server.tokens_file).consumer(token) is None


def test_bad_token_is_explained(run):
    code, _, err = run("--token", "vs-wrong", "voices")
    assert code == 1 and "HTTP 401" in err and "token was refused" in err


def test_voices_and_config(run):
    code, out, _ = run("voices")
    assert code == 0 and "other languages go to quality" in out and "en=af_heart" in out
    code, out, _ = run("config")
    assert code == 0 and "TTS fast" in out and "fallback quality" in out


def test_models_list_shows_unused_downloads(run):
    code, out, _ = run("models")
    assert code == 0 and "STT whisper-turbo (default)  [loaded]" in out
    assert "Downloaded but not used" in out and "fake/old-candidate" in out


def test_models_add_and_remove(run, cfg, monkeypatch, isolated_models):
    import huggingface_hub
    from types import SimpleNamespace
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(model_info=lambda repo, expand=None: SimpleNamespace(
        pipeline_tag="text-to-speech", tags=["license:mit"])))
    code, out, _ = run("models", "add", "newvoice", "fake/new-tts", "--languages", "en,fr", "--voice", "anna")
    assert code == 0 and isolated_models["synced"] == 1
    local = cfg.source[0].with_name("config.local.toml").read_text()
    assert "[tts.models.newvoice]" in local and 'speaker = "anna"' in local and "licence: mit" in local
    from voiceserver.config import load_config
    m = load_config(cfg.source[0]).models["newvoice"]
    assert m.languages == ["en", "fr"] and not m.preload and m.idle_unload_s == 600
    code, _, err = run("models", "add", "newvoice", "fake/other")
    assert code == 1 and "already" in err
    code, out, _ = run("models", "remove", "newvoice", "--yes")
    assert code == 0 and "newvoice" not in cfg.source[0].with_name("config.local.toml").read_text()


def test_models_remove_refuses_config_toml_models_and_used_ones(run):
    code, _, err = run("models", "remove", "parakeet", "--yes")
    assert code == 1 and "config.toml" in err
    code, _, err = run("models", "remove", "quality", "--yes")
    assert code == 1 and "still used" in err


def test_models_delete_and_clean(run, isolated_models):
    code, _, err = run("models", "delete", "fake/whisper-large-v3-turbo", "--yes")
    assert code == 1 and "used by the configuration" in err
    code, out, _ = run("models", "clean", "--yes")
    assert code == 0 and isolated_models["deleted"] == ["fake/old-candidate"] and "freed 2.0 GB" in out
    code, out, _ = run("models", "clean", "--yes")
    assert "nothing to clean" in out


@pytest.mark.parametrize("argv,script,extra", [
    (["start"], "voice-server.sh", ["start"]), (["stop"], "voice-server.sh", ["stop"]),
    (["restart"], "voice-server.sh", ["restart"]), (["check"], "health-check.sh", []),
    (["logs"], "voice-server.sh", ["logs", "50"]), (["logs", "-n", "200"], "voice-server.sh", ["logs", "200"]),
    (["install", "--no-start"], "install.sh", ["--no-start"]), (["upgrade", "--check"], "upgrade.sh", ["--check"]),
    (["uninstall", "--purge"], "uninstall.sh", ["--purge"]), (["uninstall"], "uninstall.sh", []),
])
def test_service_commands_call_their_scripts(monkeypatch, cfg, argv, script, extra):
    calls = []
    monkeypatch.setattr(cli, "run_script", lambda name, *a: calls.append((name, list(a))) or 0)
    assert cli.main([*argv, "--config", str(cfg.source[0])]) == 0
    assert calls == [(script, extra)]
