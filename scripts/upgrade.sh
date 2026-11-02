#!/bin/bash
# Upgrades are explicit. Nothing here runs on a schedule.
#
#   scripts/upgrade.sh --check            list pinned packages that have newer releases (changes nothing)
#   scripts/upgrade.sh                    apply the current uv.lock (after `git pull` or an edited pin),
#                                         fetch any new models, restart, check health
#
# To move a pin: edit the version in pyproject.toml, run `.tools/bin/uv lock` (or
# `uv lock --upgrade-package mlx-audio`), review the uv.lock diff, run tests, then this script.
# Roll back: `git checkout <previous> -- pyproject.toml uv.lock` and run this script again.
set -euo pipefail
source "$(dirname "$0")/common.sh"
UV="$(uv_bin)"
cd "$PROJECT"

if [ "${1:-}" = "--check" ]; then
  "$UV" pip list --outdated --python "$PROJECT/.venv/bin/python" 2>/dev/null \
    | grep -iE '^(Package|-|mlx|mlx-audio|mlx-metal|fastapi|uvicorn|starlette|transformers|huggingface|numpy|python-multipart)' || true
  exit 0
fi

say "Applying uv.lock"
"$UV" sync --frozen --all-groups --quiet
"$PROJECT/.venv/bin/voice-server" --check-config
say "Models:"
"$PROJECT/.venv/bin/voice-server" --prefetch
"$(dirname "$0")/voice-server.sh" restart
"$(dirname "$0")/health-check.sh"
