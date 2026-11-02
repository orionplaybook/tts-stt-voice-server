"""models.lock.json: the exact Hugging Face commit of every model the service uses.

Python packages are pinned by uv.lock; this file does the same for model files. The service only
loads the pinned commit (from the local cache, offline). Install fetches the pinned commits;
an update moves each repo to its newest commit that is at least `cooldown_days` old, so a broken
or tampered upload has time to be noticed upstream before it reaches this machine.

Some models load helper repos by name (Chatterbox loads its speech tokenizer from a second repo).
Those are listed as extra_repos in the config, pinned here too, and the cache's "main" ref of
every repo is pointed at the pinned commit, so a load by name also gets the pinned files.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from .config import PROJECT_DIR, Config

LOCK_PATH = PROJECT_DIR / "models.lock.json"


def required_repos(cfg: Config) -> list[str]:
    repos: list[str] = []
    for m in cfg.models.values():
        for r in [m.repo, *m.extra_repos]:
            if r not in repos:
                repos.append(r)
    return repos


def load(path: Path = LOCK_PATH) -> dict[str, dict]:
    return json.loads(path.read_text()).get("repos", {}) if path.exists() else {}


def save(repos: dict[str, dict], path: Path = LOCK_PATH) -> None:
    path.write_text(json.dumps({"note": "Pinned model commits; change with `voicectl update`.",
                                "repos": dict(sorted(repos.items()))}, indent=1) + "\n")


def revision(repo: str) -> str | None:
    return load().get(repo, {}).get("revision")


def _newest_settled_commit(repo: str, cooldown_days: float) -> tuple[str, str]:
    """(commit id, date) of the newest commit at least cooldown_days old; the oldest commit if
    the repo is younger than that."""
    from huggingface_hub import HfApi
    commits = HfApi().list_repo_commits(repo)  # newest first
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=cooldown_days)
    chosen = next((c for c in commits if c.created_at <= cutoff), commits[-1])
    return chosen.commit_id, chosen.created_at.strftime("%Y-%m-%d")


def _cached_revision(repo: str) -> str | None:
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(repo, local_files_only=True)).name
    except Exception:
        return None


def _commit_date(repo: str, sha: str) -> str:
    try:
        from huggingface_hub import HfApi
        return next((c.created_at.strftime("%Y-%m-%d") for c in HfApi().list_repo_commits(repo)
                     if c.commit_id == sha), "")
    except Exception:
        return ""


def _point_main_at(repo: str, sha: str) -> None:
    from huggingface_hub import constants
    refs = Path(constants.HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(sha)


def sync(cfg: Config, update: bool = False, cooldown_days: float = 7, quiet: bool = False) -> list[dict]:
    """Make sure every required repo is in the cache at its pinned commit. With update=True, move
    pins to the newest settled commit first. Returns the changes [{repo, old, new, date}]."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()  # one line per model instead (works even if huggingface_hub was imported first)

    locked = load()
    changes = []
    for repo in required_repos(cfg):
        entry = locked.get(repo)
        if entry is None and not update and (cached := _cached_revision(repo)):
            # First pin of a model already on this machine: pin what is here (what was tested).
            entry = {"revision": cached, "date": _commit_date(repo, cached)}
            changes.append({"repo": repo, "old": None, "new": cached, "date": entry["date"]})
        if update or entry is None:
            sha, date = _newest_settled_commit(repo, cooldown_days)
            if entry is None or entry["revision"] != sha:
                changes.append({"repo": repo, "old": entry["revision"] if entry else None, "new": sha, "date": date})
                entry = {"revision": sha, "date": date}
        if not quiet:
            print(f"  {repo} @ {entry['revision'][:10]} ({entry['date']}) ...", end=" ", flush=True)
        path = Path(snapshot_download(repo, revision=entry["revision"]))
        record_download(repo)
        _point_main_at(repo, entry["revision"])
        locked[repo] = entry
        if not quiet:
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            print(f"ok ({size / 1e9:.1f} GB)", flush=True)
    # Drop repos no longer configured.
    for repo in [r for r in locked if r not in required_repos(cfg)]:
        changes.append({"repo": repo, "old": locked.pop(repo)["revision"], "new": None, "date": None})
    save(locked)
    return changes


def pin_main_refs() -> None:
    """Point every pinned repo's cache "main" ref at its pinned commit (after a rollback)."""
    for repo, entry in load().items():
        _point_main_at(repo, entry["revision"])


# --- downloads made by this project (so cleaning never touches other tools' cache entries) ---

REGISTRY = Path.home() / "Library" / "Application Support" / "voice-server" / "downloaded.json"


def record_download(repo: str) -> None:
    try:
        REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        known = set(json.loads(REGISTRY.read_text())) if REGISTRY.exists() else set()
        if repo not in known:
            REGISTRY.write_text(json.dumps(sorted(known | {repo}), indent=1) + "\n")
    except Exception:
        pass


def downloaded() -> set[str]:
    return set(json.loads(REGISTRY.read_text())) if REGISTRY.exists() else set()


def forget_download(repo: str) -> None:
    known = downloaded() - {repo}
    if REGISTRY.exists():
        REGISTRY.write_text(json.dumps(sorted(known), indent=1) + "\n")


def cached_repos() -> dict[str, int]:
    """repo -> bytes on disk, for everything in the Hugging Face cache."""
    from huggingface_hub import scan_cache_dir
    try:
        return {r.repo_id: r.size_on_disk for r in scan_cache_dir().repos}
    except Exception:
        return {}


def delete_cached(repo: str) -> int:
    """Delete every cached revision of a repo; returns bytes freed."""
    from huggingface_hub import scan_cache_dir
    cache = scan_cache_dir()
    hashes = [rev.commit_hash for r in cache.repos if r.repo_id == repo for rev in r.revisions]
    if not hashes:
        return 0
    strategy = cache.delete_revisions(*hashes)
    strategy.execute()
    forget_download(repo)
    return strategy.expected_freed_size
