#!/bin/bash
# Operate the launchd agent:  scripts/voice-server.sh start|stop|restart|status|logs|check
set -euo pipefail
source "$(dirname "$0")/common.sh"

domain="gui/$(id -u)"
rotate_launchd_log() {  # launchd's own stdout/stderr file only holds crash output; cap it anyway
  local f="$LOG_DIR/launchd.log"
  if [ -f "$f" ] && [ "$(stat -f %z "$f")" -gt 5242880 ]; then mv -f "$f" "$f.1"; fi
}

case "${1:-status}" in
  start)
    [ -f "$PLIST" ] || die "not installed; run scripts/install.sh"
    rotate_launchd_log
    launchctl bootstrap "$domain" "$PLIST" 2>/dev/null || launchctl kickstart "$domain/$LABEL"
    wait_healthy 300 || die "not healthy; see $LOG_DIR/voice-server.log" ;;
  stop)
    launchctl bootout "$domain/$LABEL" 2>/dev/null && say "stopped" || say "not running" ;;
  restart)
    rotate_launchd_log
    if launchctl print "$domain/$LABEL" >/dev/null 2>&1; then
      launchctl kickstart -k "$domain/$LABEL"
    else
      launchctl bootstrap "$domain" "$PLIST"
    fi
    sleep 2
    wait_healthy 300 || die "not healthy; see $LOG_DIR/voice-server.log" ;;
  status)
    launchctl print "$domain/$LABEL" 2>/dev/null | grep -E $'^\t(state|pid|last exit code|runs) =' || say "agent not loaded"
    url="$(server_url)"
    probe -fsS --max-time 3 "$url/health" && echo || say "health: not answering on $url" ;;
  logs)
    tail -n "${2:-50}" -F "$LOG_DIR/voice-server.log" ;;
  check)
    exec "$(dirname "$0")/health-check.sh" ;;
  *)
    die "usage: $0 start|stop|restart|status|logs [n]|check" ;;
esac
