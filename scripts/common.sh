# Shared settings for the scripts in this folder. Sourced, not run.
# Everything the service needs lives in this project folder: uv (.tools), Python (.python),
# the virtualenv (.venv) and uv's cache (.uv-cache). Models go to the Hugging Face cache
# (~/.cache/huggingface), shared with other tools that use it.

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${VOICE_SERVER_LABEL:-local.voice-server}"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="${VOICE_SERVER_LOG_DIR:-$HOME/Library/Logs/voice-server}"
UV_VERSION="0.13.6"
export UV_PYTHON_INSTALL_DIR="$PROJECT/.python"
export UV_CACHE_DIR="$PROJECT/.uv-cache"
export UV_PYTHON_PREFERENCE="only-managed"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

say()  { printf '%s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

uv_bin() {
  # A uv already on PATH is fine if it is the pinned version; otherwise use (or create) the
  # project-local copy, installed with pip into .tools from the system Python. Nothing system-wide.
  if command -v uv >/dev/null 2>&1 && [ "$(uv --version | awk '{print $2}')" = "$UV_VERSION" ]; then
    command -v uv; return
  fi
  if [ ! -x "$PROJECT/.tools/bin/uv" ] || [ "$("$PROJECT/.tools/bin/uv" --version | awk '{print $2}')" != "$UV_VERSION" ]; then
    say "Setting up uv $UV_VERSION in $PROJECT/.tools" >&2
    /usr/bin/python3 -m venv "$PROJECT/.tools" >&2
    "$PROJECT/.tools/bin/pip" install -q --disable-pip-version-check "uv==$UV_VERSION" >&2
  fi
  printf '%s\n' "$PROJECT/.tools/bin/uv"
}

# Base URL from the config (config.local.toml overrides config.toml), http or https.
server_url() {
  "$PROJECT/.venv/bin/python" - "$PROJECT" <<'PY'
import sys
sys.path.insert(0, sys.argv[1] + "/src")
from voiceserver.config import base_url, load_config
print(base_url(load_config().server))
PY
}

# curl for unauthenticated local probes; -k because a local HTTPS certificate is issued for the
# Mac's name, not 127.0.0.1 (no token is ever sent by these probes).
probe() { curl -k "$@"; }

wait_healthy() {  # seconds
  local url; url="$(server_url)"
  for _ in $(seq 1 "${1:-300}"); do
    if probe -fsS --max-time 2 "$url/health" >/dev/null 2>&1; then
      probe -fsS "$url/health"; echo; return 0
    fi
    sleep 1
  done
  return 1
}
