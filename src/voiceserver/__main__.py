"""Entry point: `voice-server [--config FILE] [--host H] [--port P] [--debug-content]`."""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys

import uvicorn

from .app import create_app
from .config import is_loopback, load_config


def setup_logging(log_dir, debug_content: bool) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(log_dir / "voice-server.log", maxBytes=10 * 2**20,
                                                   backupCount=5, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    if sys.stderr.isatty():
        root.addHandler(logging.StreamHandler())
    root.setLevel(logging.INFO)
    logging.getLogger("voiceserver").setLevel(logging.DEBUG if debug_content else logging.INFO)
    # Libraries can log request details or model text; keep them at warnings.
    for noisy in ("uvicorn.access", "httpx", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="voice-server")
    ap.add_argument("--config")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--debug-content", action="store_true",
                    help="log transcripts and speech input (never in normal operation)")
    ap.add_argument("--prefetch", action="store_true",
                    help="download the pinned commit of every configured model (models.lock.json), then exit")
    ap.add_argument("--check-config", action="store_true", help="validate the config, then exit")
    args = ap.parse_args(argv)

    if not args.prefetch:
        # Serve from the local Hugging Face cache only: models are fetched by --prefetch (install
        # and upgrade scripts), never by a running service.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    cfg = load_config(args.config)
    if args.check_config:
        print(f"config OK: {', '.join(str(p) for p in cfg.source)}")
        for m in cfg.models.values():
            print(f"  {m.kind} {m.name}: {m.repo}{' (preload)' if m.preload else ''}")
        return
    if args.prefetch:
        from . import modellock
        modellock.sync(cfg)  # fetch the commits pinned in models.lock.json (pins new repos)
        return
    if args.host:
        cfg.server.host = args.host
    if args.port:
        cfg.server.port = args.port
    if args.debug_content:
        cfg.server.log_content = True
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    setup_logging(cfg.server.log_dir, cfg.server.log_content)
    logging.getLogger("voiceserver").info("config: %s", ", ".join(str(p) for p in cfg.source))
    s = cfg.server
    tls = bool(s.tls_cert and s.tls_key)
    if not is_loopback(s.host) and not tls and not s.allow_plain_http_remote:
        # Tokens and audio would cross the network in clear text.
        sys.exit(f"voice-server: refusing to listen on {s.host} over plain HTTP. Configure tls_cert and "
                 "tls_key (HTTPS), or keep host = 127.0.0.1 and put a TLS reverse proxy in front, or set "
                 "allow_plain_http_remote = true if you accept that risk on a trusted network.")
    uvicorn.run(create_app(cfg), host=s.host, port=s.port, log_config=None, access_log=False,
                server_header=False, timeout_graceful_shutdown=10,
                ssl_certfile=s.tls_cert if tls else None, ssl_keyfile=s.tls_key if tls else None)


if __name__ == "__main__":
    main()
