#!/bin/bash
# Remove the launchd agent.
#   scripts/uninstall.sh                 stop and remove the agent; keep tokens, logs, models, venv
#   scripts/uninstall.sh --purge         also delete the venv/uv/Python in this folder, the logs,
#                                        the token file and the configured models from the HF cache
set -euo pipefail
source "$(dirname "$0")/common.sh"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null && say "stopped $LABEL" || true
rm -f "$PLIST" && say "removed $PLIST"

if [ "${1:-}" = "--purge" ]; then
  if [ -x "$PROJECT/.venv/bin/python" ]; then
    "$PROJECT/.venv/bin/python" - "$PROJECT" <<'PY'
import shutil, sys
sys.path.insert(0, sys.argv[1] + "/src")
from huggingface_hub import scan_cache_dir
from voiceserver.config import load_config
cfg = load_config()
repos = {m.repo for m in cfg.models.values()}
try:
    cache = scan_cache_dir()
    hashes = [rev.commit_hash for r in cache.repos if r.repo_id in repos for rev in r.revisions]
    if hashes:
        strategy = cache.delete_revisions(*hashes)
        print(f"deleting {len(hashes)} model revisions ({strategy.expected_freed_size_str})")
        strategy.execute()
except Exception as e:
    print(f"model cache: {e}")
tokens = cfg.server.tokens_file
if tokens.exists():
    tokens.unlink()
    print(f"deleted {tokens}")
PY
  fi
  rm -rf "$PROJECT/.venv" "$PROJECT/.tools" "$PROJECT/.python" "$PROJECT/.uv-cache" "$LOG_DIR"
  say "purged environment and logs"
fi
