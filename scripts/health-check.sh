#!/bin/bash
# Health and auth check, suitable for a nightly job. Exit 0 = all good.
#   scripts/health-check.sh [base-url]        default: from the config, e.g. http://127.0.0.1:8100
# Prints one "ok|fail <check> <detail>" line per check.
set -uo pipefail
source "$(dirname "$0")/common.sh"

BASE="${1:-$(server_url)}"
rc=0
check() { printf '%s\t%s\t%s\n' "$1" "$2" "$3"; [ "$1" = ok ] || rc=1; }

body="$(probe -sS --max-time 10 -w '\n%{http_code}' "$BASE/health" 2>&1)"
code="${body##*$'\n'}"; body="${body%$'\n'*}"
if [ "$code" = 200 ] && printf '%s' "$body" | grep -q '"status":"ok"'; then
  check ok voice.health "HTTP 200, loaded: $(printf '%s' "$body" | sed -E 's/.*"models":\[([^]]*)\].*/\1/')"
else
  check fail voice.health "HTTP ${code:-none} (${body:0:200})"
fi

for path in /v1/models; do
  code="$(probe -s -o /dev/null -w '%{http_code}' --max-time 10 "$BASE$path")"
  [ "$code" = 401 ] && check ok voice.token-required "HTTP 401 without a token on $path" \
                    || check fail voice.token-required "HTTP $code without a token on $path (expected 401)"
done
code="$(probe -s -o /dev/null -w '%{http_code}' --max-time 10 -X POST -H 'Content-Type: application/json' \
        -d '{"input":"x"}' "$BASE/v1/audio/speech")"
[ "$code" = 401 ] && check ok voice.speech-token-required "HTTP 401 without a token" \
                  || check fail voice.speech-token-required "HTTP $code without a token (expected 401)"
exit $rc
