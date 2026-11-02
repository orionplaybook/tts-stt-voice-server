"""Bearer tokens, one per consumer, stored hashed.

The file holds {"tokens": [{"name", "sha256", "created"}]} and is written with mode 600. A token
is 32 random bytes (urlsafe base64, prefixed "vs-"), so a plain SHA-256 is enough to store it:
there is nothing to brute-force, unlike a password. Lookups compare digests in constant time.

CLI (the service re-reads the file when it changes, no restart needed):
  voice-tokens add <name>       create a token for a consumer, print it once
  voice-tokens list             consumer names and creation dates (never tokens)
  voice-tokens revoke <name>    delete that consumer's token
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import tempfile
import threading
from pathlib import Path

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class TokenStore:
    def __init__(self, path: Path):
        self.path = Path(path).expanduser()
        self._lock = threading.Lock()
        self._mtime: float | None = None
        self._entries: list[dict] = []

    # --- reading (service side) ---------------------------------------------------------

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text())
        return list(data.get("tokens", []))

    def _refresh(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            self._entries, self._mtime = [], None
            return
        if mtime != self._mtime:
            self._entries, self._mtime = self._load(), mtime

    def consumer(self, token: str | None) -> str | None:
        """The consumer name for a presented token, or None."""
        if not token:
            return None
        digest = _digest(token)
        with self._lock:
            self._refresh()
            match = None
            for entry in self._entries:  # no early exit: same work for every presented token
                if hmac.compare_digest(entry.get("sha256", ""), digest):
                    match = entry.get("name")
            return match

    def names(self) -> list[dict]:
        with self._lock:
            self._refresh()
            return [{"name": e["name"], "created": e.get("created", "")} for e in self._entries]

    def count(self) -> int:
        return len(self.names())

    # --- writing (CLI side) -------------------------------------------------------------

    def _save(self, entries: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".tokens-")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump({"tokens": entries}, f, indent=1)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def add(self, name: str) -> str:
        if not NAME_RE.match(name):
            raise ValueError("name: lowercase letters, digits, '.', '_' or '-', at most 64 characters")
        entries = self._load()
        if any(e["name"] == name for e in entries):
            raise ValueError(f"a token named {name!r} already exists; revoke it first to replace it")
        token = "vs-" + secrets.token_urlsafe(32)
        entries.append({"name": name, "sha256": _digest(token),
                        "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
        self._save(entries)
        return token

    def revoke(self, name: str) -> bool:
        entries = self._load()
        kept = [e for e in entries if e["name"] != name]
        if len(kept) == len(entries):
            return False
        self._save(kept)
        return True

    def check_permissions(self) -> str | None:
        """A warning if the file is readable by others."""
        if self.path.exists() and self.path.stat().st_mode & 0o077:
            return f"{self.path} is readable by other users; run: chmod 600 '{self.path}'"
        return None


def main(argv: list[str] | None = None) -> int:
    from .config import load_config

    ap = argparse.ArgumentParser(prog="voice-tokens", description="Manage voice-server consumer tokens.")
    ap.add_argument("--config", help="config file (default: config.toml next to the project)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("add", help="create a token for a consumer").add_argument("name")
    sub.add_parser("list", help="list consumer names")
    sub.add_parser("revoke", help="revoke a consumer's token").add_argument("name")
    args = ap.parse_args(argv)

    store = TokenStore(load_config(args.config).server.tokens_file)
    if args.cmd == "add":
        try:
            token = store.add(args.name)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"Token for {args.name!r} (shown once, store it with the consumer):\n{token}")
    elif args.cmd == "list":
        names = store.names()
        for e in names:
            print(f"{e['name']}\t{e['created']}")
        if not names:
            print(f"no tokens in {store.path}")
    elif args.cmd == "revoke":
        if not store.revoke(args.name):
            print(f"error: no token named {args.name!r}", file=sys.stderr)
            return 1
        print(f"revoked {args.name!r}")
    if warning := store.check_permissions():
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
