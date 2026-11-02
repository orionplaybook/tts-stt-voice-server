#!/bin/bash
# Install voice-server as a launchd user agent (starts at login, restarted if it dies).
#
#   scripts/install.sh            set up the environment from uv.lock, fetch models, start
#   scripts/install.sh --no-start set everything up but leave the agent unloaded
#
# Needs: Apple Silicon Mac, ffmpeg on PATH (e.g. `brew install ffmpeg`). Installs nothing
# system-wide: uv, Python and the virtualenv stay in this folder; the agent plist goes to
# ~/Library/LaunchAgents. Safe to re-run.
set -euo pipefail
source "$(dirname "$0")/common.sh"

START=1
[ "${1:-}" = "--no-start" ] && START=0

[ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ] || die "Apple Silicon macOS required (MLX)"
command -v ffmpeg >/dev/null || die "ffmpeg not found. Install it (for example: brew install ffmpeg), then re-run."
[ -f "$PROJECT/uv.lock" ] || die "uv.lock missing: the environment is installed only from the pinned lockfile"

UV="$(uv_bin)"
say "Installing the pinned environment (uv.lock)"
(cd "$PROJECT" && "$UV" sync --frozen --all-groups --quiet)
"$PROJECT/.venv/bin/voice-server" --check-config

say "Models (downloaded on the first run, about 2 GB):"
"$PROJECT/.venv/bin/voice-server" --prefetch

mkdir -p "$LOG_DIR" "$HOME/Library/LaunchAgents"
TOKENS="$("$PROJECT/.venv/bin/python" -c "import sys; sys.path.insert(0,'$PROJECT/src'); from voiceserver.config import load_config; print(load_config().server.tokens_file)")"
mkdir -p "$(dirname "$TOKENS")" && chmod 700 "$(dirname "$TOKENS")"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>$PROJECT/.venv/bin/voice-server</string></array>
  <key>WorkingDirectory</key><string>$PROJECT</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    <key>HF_HUB_OFFLINE</key><string>1</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>$LOG_DIR/launchd.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/launchd.log</string>
</dict>
</plist>
EOF
plutil -lint "$PLIST" >/dev/null
say "Wrote $PLIST"

if [ "$START" -eq 1 ]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$PLIST"
  say "Started $LABEL; waiting for the models to load..."
  wait_healthy 300 || die "not healthy after 5 minutes; see $LOG_DIR/voice-server.log and $LOG_DIR/launchd.log"
fi
say
say "Next: create a token per consumer, e.g.  $PROJECT/.venv/bin/voice-tokens add my-agent"
