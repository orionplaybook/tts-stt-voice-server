#!/bin/bash
# Check what git would commit from this project (tracked files plus new files not ignored):
# no tokens, keys or token stores, no third-party libraries, no model weights, no large binaries.
# Third-party code and models are downloaded at install time (uv.lock, models.lock.json), never
# stored in the repository.
#
#   scripts/check-repo.sh            check this project folder (run before every commit)
#   scripts/check-repo.sh DIR        check another folder (the public export uses this)
# Exit 0 = clean. Prints one line per check.
set -uo pipefail

DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$DIR" || exit 2
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  FILES="$(git ls-files --cached --others --exclude-standard -- . | sort -u)"
else
  FILES="$(find . -type f -not -path './.git/*' | sed 's#^\./##' | sort)"
fi
# report() runs at the end of pipelines (a subshell): record failures in a file, not a variable.
FAILED="$(mktemp)"; rm -f "$FAILED"; trap 'rm -f "$FAILED"' EXIT
report() {  # title, findings on stdin
  local out; out="$(cat)"
  if [ -n "$out" ]; then echo "! $1:"; printf '%s\n' "$out" | sed 's/^/    /' | cut -c1-200; : > "$FAILED"
  else echo "  ok  $1"; fi
}
files() { printf '%s\n' "$FILES" | grep -v '^$'; }

files | grep -E '(^|/)(\.venv|venv|\.tools|\.python|\.uv-cache|site-packages|node_modules|__pycache__|[^/]+\.dist-info|[^/]+\.egg-info)(/|$)' \
  | report "no virtualenvs or installed third-party packages"
files | grep -E '\.(whl|so|dylib|dll|a|o|tar\.gz|tgz|zip|jar)$' | report "no wheels, archives or compiled libraries"
files | grep -E '\.(safetensors|gguf|ggml|bin|pt|pth|ckpt|onnx|npz|npy|mlmodel|mlpackage|h5|tflite)$' \
  | report "no model weights"
files | while IFS= read -r f; do
  [ -f "$f" ] && [ "$(stat -f %z "$f")" -gt 5242880 ] && printf '%s (%s MB)\n' "$f" "$(( $(stat -f %z "$f") / 1048576 ))"
done | report "no file larger than 5 MB"
files | grep -E -i '(^|/)(tokens\.json|[^/]*\.token|\.env|[^/]*\.env|[^/]*secret[^/]*|[^/]*\.pem|[^/]*\.key|id_rsa[^/]*|\.netrc)$' \
  | report "no token stores, env files or keys"
# Built from pieces so that this file does not match its own patterns.
SECRETS="(vs-[A-Za-z0-9_-]{40,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|xox[abp]-[A-Za-z0-9-]{10,}|gh""p_[A-Za-z0-9]{20,}"
SECRETS="$SECRETS|github""_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIV""ATE KEY|\"sha256\": *\"[0-9a-f]{64}\")"
files | while IFS= read -r f; do
  [ -f "$f" ] && grep -I -n -E "$SECRETS" "$f" 2>/dev/null | sed "s#^#$f:#"
done | report "no tokens, API keys, private keys or token hashes in file contents"

if [ -e "$FAILED" ]; then echo "repository check FAILED"; exit 1; fi
echo "repository check passed: $(files | wc -l | tr -d ' ') files"
