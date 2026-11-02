import json
import stat

import pytest

from voiceserver.tokens import TokenStore, main


def test_add_stores_only_a_hash_with_private_mode(tmp_path):
    store = TokenStore(tmp_path / "p" / "tokens.json")
    token = store.add("agent-one")
    raw = store.path.read_text()
    assert token not in raw
    entry = json.loads(raw)["tokens"][0]
    assert entry["name"] == "agent-one" and len(entry["sha256"]) == 64
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.path.parent.stat().st_mode) == 0o700
    assert store.consumer(token) == "agent-one"


def test_wrong_or_missing_token_is_rejected(tmp_path):
    store = TokenStore(tmp_path / "tokens.json")
    token = store.add("n8n")
    assert store.consumer(token + "x") is None
    assert store.consumer("") is None
    assert store.consumer(None) is None


def test_one_token_per_consumer_and_revoke(tmp_path):
    store = TokenStore(tmp_path / "tokens.json")
    a, b = store.add("a"), store.add("b")
    with pytest.raises(ValueError):
        store.add("a")
    assert [e["name"] for e in store.names()] == ["a", "b"]
    assert store.revoke("a") and not store.revoke("a")
    assert store.consumer(a) is None and store.consumer(b) == "b"


def test_bad_names_refused(tmp_path):
    store = TokenStore(tmp_path / "tokens.json")
    for bad in ("", "Has Space", "../x", "UPPER", "x" * 65):
        with pytest.raises(ValueError):
            store.add(bad)


def test_service_sees_changes_without_restart(tmp_path):
    reader = TokenStore(tmp_path / "tokens.json")
    writer = TokenStore(tmp_path / "tokens.json")
    assert reader.consumer("vs-nope") is None
    token = writer.add("late")
    assert reader.consumer(token) == "late"
    writer.revoke("late")
    assert reader.consumer(token) is None


def test_cli_list_never_prints_tokens(tmp_path, capsys, cfg):
    store = TokenStore(cfg.server.tokens_file)
    token = store.add("hermes-x")
    assert main(["--config", str(cfg.source[0]), "list"]) == 0
    out = capsys.readouterr().out
    assert "hermes-x" in out and token not in out and store.names()[0]["created"] in out


def test_http_requires_token(cfg, token, fake_engines):
    from fastapi.testclient import TestClient

    from voiceserver.app import create_app
    with TestClient(create_app(cfg)) as c:
        assert c.get("/health").status_code == 200
        r = c.get("/v1/models")
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
        assert r.json()["error"]["code"] == "invalid_api_key"
        assert c.get("/v1/models", headers={"Authorization": "Bearer vs-wrong"}).status_code == 401
        assert c.get("/v1/models", headers={"Authorization": token}).status_code == 401
        assert c.post("/v1/audio/speech", json={"input": "hi"}).status_code == 401
        assert c.get("/v1/models", headers={"Authorization": f"Bearer {token}"}).status_code == 200


def test_logs_name_the_consumer_never_the_token(cfg, token, fake_engines, caplog):
    import logging

    from fastapi.testclient import TestClient

    from voiceserver.app import create_app
    caplog.set_level(logging.INFO, logger="voiceserver")
    with TestClient(create_app(cfg)) as c:
        c.post("/v1/audio/speech", json={"input": "Hello there, this is a test."},
               headers={"Authorization": f"Bearer {token}"})
        c.get("/v1/models", headers={"Authorization": "Bearer vs-wrong-token"})
    text = caplog.text
    assert "consumer=tester" in text
    assert token not in text and "vs-wrong-token" not in text
    assert "Hello there" not in text  # content is not logged by default


def test_realtime_requires_token(cfg, token, fake_engines):
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from voiceserver.app import create_app
    with TestClient(create_app(cfg)) as c:
        with pytest.raises(WebSocketDisconnect) as e:
            with c.websocket_connect("/v1/realtime?intent=transcription") as ws:
                ws.receive_json()
        assert e.value.code == 1008
