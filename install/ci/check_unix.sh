#!/usr/bin/env bash
# CI assertions for the macOS/Linux installer. Usage: check_unix.sh {shortcuts|run|ai|open-bundle|uninstall}
set -euo pipefail
MODE="${1:?mode}"
PH="${POLONY_HOME:-$HOME/PolonyReview}"
VPY="$PH/venv/bin/python"
case "$(uname -s)" in Darwin) OS=mac ;; *) OS=linux ;; esac
ok() { echo "OK: $*"; }
bad() { echo "FAIL: $*" >&2; exit 1; }

launcher() { "$VPY" -m polony_app.launcher "$@"; }

case "$MODE" in
shortcuts)
  [ -x "$VPY" ] || bad "venv python missing"
  [ -f "$PH/config.json" ] || bad "config.json missing"
  [ -x "$PH/bin/polony-review" ] || bad "polony-review helper missing"
  [ -f "$PH/app/src/polony_app/webdist/index.html" ] || bad "prebuilt UI missing"
  ok "install folder complete"
  if [ "$OS" = mac ]; then
    for base in "$HOME/Applications" "$HOME/Desktop"; do
      A="$base/Polony Review.app"
      [ -d "$A" ] || bad "$A missing"
      python3 - "$A" <<'PY'
import os, plistlib, sys
app = sys.argv[1]
p = plistlib.load(open(app + "/Contents/Info.plist", "rb"))
exe = f"{app}/Contents/MacOS/{p['CFBundleExecutable']}"
assert os.access(exe, os.X_OK), "launcher not executable"
icon = f"{app}/Contents/Resources/{p['CFBundleIconFile']}.icns"
assert open(icon, "rb").read(4) == b"icns", "bad icns"
print("bundle ok:", app, p["CFBundleIdentifier"])
PY
    done
    command -v xattr >/dev/null && { xattr -r "$HOME/Desktop/Polony Review.app" | grep -q quarantine && bad "quarantine attribute present" || ok "no quarantine attribute"; }
  else
    D="$HOME/Desktop/Polony Review.desktop"
    [ -x "$D" ] || bad "$D missing or not executable"
    grep -q '^Exec=' "$D" || bad "no Exec line"
    ok "desktop entry present"
  fi
  launcher --self-test
  ;;
run)
  if [ "$OS" = mac ]; then
    "$HOME/Desktop/Polony Review.app/Contents/MacOS/launcher"
  else
    "$PH/bin/polony-review"
  fi
  out="$(launcher status)"; echo "$out"
  url="$(printf '%s\n' "$out" | sed -n 's/^running: \(http[^ ]*\).*/\1/p')"
  [ -n "$url" ] || bad "server not running after launch"
  curl -fsS "${url}api/health" | grep -q '"ok":true' || bad "health failed"
  curl -fsS "$url" | grep -q '<div id="root">' || bad "UI not served at /"
  ok "launched: $url (health + UI)"
  # launching again must reuse the same server
  launcher | tail -1; [ "$(launcher status | sed -n 's/^running: \(http[^ ]*\).*/\1/p')" = "$url" ] || bad "second launch changed the server"
  launcher stop
  if launcher status >/dev/null; then bad "still running after stop"; fi
  ok "stopped"
  ;;
ai)  # only meaningful when the installer ran with POLONY_WITH_AI=1
  out="$(launcher status || true)"; echo "$out"
  printf '%s\n' "$out" | grep -q '^AI detector: installed (spotiflow' || bad "AI detector not reported as installed"
  launcher prepare-model | tail -3
  [ "$(launcher prepare-model | grep -c '^sha256: [0-9a-f]\{64\}$')" = 1 ] || bad "prepare-model gave no sha256"
  grep -q '"ai_enabled": true' "$PH/config.json" || bad "ai_enabled not recorded"
  ok "AI detector installed and model prepared"
  ;;
open-bundle)  # macOS only: LaunchServices must accept and run the bundle
  [ "$OS" = mac ] || exit 0
  open -g -j "$HOME/Desktop/Polony Review.app" --env POLONY_CI=1
  for _ in $(seq 1 90); do launcher status >/dev/null 2>&1 && break; sleep 2; done
  launcher status || bad "bundle started through 'open' did not start the server"
  launcher stop
  ok "open(1) of the bundle started the server"
  ;;
uninstall)
  bash "$PH/app/install/uninstall.sh"
  [ ! -e "$PH" ] || bad "install folder still there"
  [ ! -e "$HOME/Desktop/Polony Review.app" ] && [ ! -e "$HOME/Desktop/Polony Review.desktop" ] || bad "desktop icon still there"
  [ -d "${POLONY_DATA_DIR:-$HOME/PolonyReviewData}" ] || bad "user data was deleted"
  ok "uninstalled, data kept"
  ;;
*) bad "unknown mode $MODE" ;;
esac
