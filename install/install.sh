#!/usr/bin/env bash
# Polony Review installer for macOS and Linux. No admin rights, never uses sudo.
#
#   curl -fsSL https://raw.githubusercontent.com/oriamos/polony-counting/claude/polony-hitl-app/install/install.sh | bash
#
# Re-running it updates the installation. Options (after `bash -s --` when piped):
#   --dry-run        print what would be done, change nothing
#   --no-launch      do not start the app at the end
#   --ref NAME       git branch/tag to install (default: claude/polony-hitl-app)
#   --python PATH    use this Python (3.10-3.13) instead of searching / downloading one
# Environment: POLONY_HOME (default ~/PolonyReview), POLONY_DATA_DIR (default ~/PolonyReviewData),
#   POLONY_SOURCE (local checkout to copy instead of downloading from GitHub), POLONY_REPO, POLONY_REF,
#   POLONY_CI=1 (no browser, no prompts; used by the CI tests).
#
# What it does (each step is printed): 1) fetch the code into <home>/app (git, else a GitHub tar.gz),
# 2) find Python 3.10-3.13 or, if there is none, download `uv` (https://astral.sh/uv) and with it a
# standalone Python 3.12 into <home>, 3) create <home>/venv and pip install requirements-app.txt,
# 4) self-test, 5) create the desktop icon / app bundle.
# The whole script is wrapped in main() so a truncated download cannot run half a script.
set -euo pipefail

main() {
  REPO="${POLONY_REPO:-oriamos/polony-counting}"
  REF="${POLONY_REF:-claude/polony-hitl-app}"
  PH="${POLONY_HOME:-$HOME/PolonyReview}"
  DATA="${POLONY_DATA_DIR:-$HOME/PolonyReviewData}"
  DRY=0; NO_LAUNCH=0; PY_ARG=""; SHORTCUTS_ONLY=0
  CI="${POLONY_CI:-}"; [ "$CI" = "0" ] && CI=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --dry-run) DRY=1 ;;
      --no-launch) NO_LAUNCH=1 ;;
      --shortcuts-only) SHORTCUTS_ONLY=1; NO_LAUNCH=1 ;;   # (used by the tests) skip steps 1-4
      --ref) REF="${2:?--ref needs a value}"; shift ;;
      --python) PY_ARG="${2:?--python needs a value}"; shift ;;
      -h|--help) sed -n '2,16p' "$0" 2>/dev/null || true; return 0 ;;
      *) echo "Unknown option: $1" >&2; return 2 ;;
    esac
    shift
  done
  [ -n "$CI" ] && NO_LAUNCH=1

  case "${POLONY_FORCE_OS:-$(uname -s)}" in   # POLONY_FORCE_OS: test hook (mac|linux)
    Darwin|mac) OS=mac ;;
    Linux|linux) OS=linux ;;
    *) echo "This installer supports macOS and Linux. On Windows use install.ps1." >&2; return 1 ;;
  esac
  APP="$PH/app"; VENV="$PH/venv"; VPY="$VENV/bin/python"
  STEPS=6

  say()  { printf '%s\n' "$*"; }
  step() { printf '\n[%s/%s] %s\n' "$1" "$STEPS" "$2"; }
  warn() { printf 'WARNING: %s\n' "$*" >&2; }
  die()  { printf '\nERROR: %s\n' "$*" >&2; printf 'Nothing is broken by this: you can run the installer again.\n' >&2; exit 1; }
  plan() { printf '  [dry-run] %s\n' "$*"; }
  sq()   { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }   # shell-quote one string

  say "Polony Review installer ($OS)"
  say "  install folder : $PH"
  say "  your data      : $DATA   (never deleted by updates or the uninstaller unless you ask)"
  say "  source         : ${POLONY_SOURCE:-github.com/$REPO ($REF)}"
  [ "$DRY" = 1 ] && say "  DRY RUN: nothing will be changed."

  have() { command -v "$1" >/dev/null 2>&1; }
  mac_has_clt() { [ "$OS" != mac ] || xcode-select -p >/dev/null 2>&1; }

  install_everything() {
    # ---------------------------------------------------------------- 1. code
    step 1 "Getting the app code into $APP"
    fetch_code() {
      if [ -n "${POLONY_SOURCE:-}" ]; then
        [ -d "$POLONY_SOURCE/src/polony_app" ] || die "POLONY_SOURCE=$POLONY_SOURCE is not a Polony checkout"
        say "  copying the local checkout $POLONY_SOURCE"
        local tmp="$PH/app.new"; rm -rf "$tmp"; mkdir -p "$tmp"
        if [ -d "$POLONY_SOURCE/.git" ] && have git; then
          ( cd "$POLONY_SOURCE" && git ls-files --cached --others --exclude-standard -z | tar --null -T - -cf - ) | tar -xf - -C "$tmp"
        else
          ( cd "$POLONY_SOURCE" && tar --exclude=.git --exclude=node_modules --exclude=__pycache__ --exclude=./data -cf - . ) | tar -xf - -C "$tmp"
        fi
        rm -rf "$APP"; mv "$tmp" "$APP"
        return
      fi
      if have git && mac_has_clt; then
        if [ -d "$APP/.git" ]; then
          say "  updating the existing copy with git"
          git -C "$APP" fetch --depth 1 origin "$REF" >/dev/null 2>&1 \
            && git -C "$APP" checkout -q -f FETCH_HEAD \
            && return 0
          warn "git update failed; falling back to a fresh download"
          rm -rf "$APP"
        else
          say "  cloning https://github.com/$REPO.git ($REF)"
          rm -rf "$APP"
          git clone -q --depth 1 --branch "$REF" "https://github.com/$REPO.git" "$APP" && return 0
          warn "git clone failed; falling back to a download of the archive"
          rm -rf "$APP"
        fi
      fi
      have curl || die "curl is needed to download the app (it ships with macOS and most Linux systems)"
      local url="https://github.com/$REPO/archive/refs/heads/$REF.tar.gz"
      say "  downloading $url"
      local tmp="$PH/app.new"; rm -rf "$tmp"; mkdir -p "$tmp"
      curl -fL --progress-bar "$url" | tar -xzf - -C "$tmp" --strip-components=1 || die "download of $url failed"
      rm -rf "$APP"; mv "$tmp" "$APP"
    }
    if [ "$DRY" = 0 ] && [ -x "$VPY" ] && [ -d "$APP" ]; then
      say "  stopping a running Polony Review (if any) so the update takes effect"
      POLONY_HOME="$PH" "$VPY" -m polony_app.launcher stop >/dev/null 2>&1 || true
    fi
    if [ "$DRY" = 1 ]; then plan "stop a running Polony Review; mkdir -p $PH; fetch code -> $APP"; else mkdir -p "$PH"; fetch_code; fi
    [ "$DRY" = 1 ] || [ -f "$APP/requirements-app.txt" ] || die "the downloaded code has no requirements-app.txt"

    # ---------------------------------------------------------------- 2. python + venv
    step 2 "Preparing Python (3.10-3.13) and a private virtual environment in $VENV"
    py_ok() {  # $1 = interpreter; prints its version when it is 3.10-3.13
      "$1" -c 'import sys; v=sys.version_info[:2]; assert (3,10)<=v<=(3,13); print("%d.%d"%v)' 2>/dev/null
    }
    find_python() {
      local c p
      [ -z "${POLONY_FORCE_UV:-}" ] || return 1   # test hook: exercise the "download Python with uv" path
      for c in ${PY_ARG:+"$PY_ARG"} python3.12 python3.11 python3.13 python3.10 python3 python; do
        p="$(command -v "$c" 2>/dev/null || true)"; [ -n "$p" ] || continue
        # the macOS /usr/bin stubs open a "install developer tools" dialog when the tools are missing
        case "$p" in /usr/bin/python*) mac_has_clt || continue ;; esac
        if py_ok "$p" >/dev/null; then printf '%s' "$p"; return 0; fi
      done
      return 1
    }
    make_venv_with() {  # $1 = python; True when a working venv with pip exists afterwards
      rm -rf "$VENV"
      "$1" -m venv "$VENV" >/dev/null 2>&1 && "$VPY" -m pip --version >/dev/null 2>&1
    }
    bootstrap_uv() {
      UVBIN="$PH/uv-bin"
      if [ -x "$UVBIN/uv" ]; then UV="$UVBIN/uv"; return; fi
      if have uv; then UV="$(command -v uv)"; return; fi
      have curl || die "curl is needed to download uv/Python"
      say "  No suitable Python found. Downloading the small 'uv' tool from https://astral.sh/uv/install.sh"
      say "  (it is installed only inside $UVBIN; your PATH and shell files are not modified)."
      mkdir -p "$UVBIN"
      curl -LsSf https://astral.sh/uv/install.sh -o "$PH/uv-install.sh" || die "could not download the uv installer"
      UV_UNMANAGED_INSTALL="$UVBIN" UV_NO_MODIFY_PATH=1 sh "$PH/uv-install.sh" >/dev/null 2>&1 || die "the uv installer failed"
      rm -f "$PH/uv-install.sh"
      UV="$UVBIN/uv"; [ -x "$UV" ] || die "uv was not installed"
    }
    setup_venv() {
      if [ -x "$VPY" ] && py_ok "$VPY" >/dev/null && "$VPY" -m pip --version >/dev/null 2>&1; then
        say "  reusing the existing environment ($(py_ok "$VPY"))"; return
      fi
      local py
      if py="$(find_python)" && make_venv_with "$py"; then
        say "  using the Python $(py_ok "$py") found at $py"; return
      fi
      [ -n "${py:-}" ] && warn "could not create a virtual environment with $py (on Debian/Ubuntu 'python3-venv' is missing); using a downloaded Python instead"
      bootstrap_uv
      say "  Downloading a standalone Python 3.12 with uv (about 30 MB) ..."
      rm -rf "$VENV"
      UV_PYTHON_INSTALL_DIR="$PH/python" "$UV" venv --seed --python 3.12 "$VENV" || die "uv could not set up Python 3.12"
      "$VPY" -m pip --version >/dev/null 2>&1 || die "the virtual environment has no pip"
    }
    if [ "$DRY" = 1 ]; then
      if p="$(find_python)"; then plan "use existing Python at $p -> venv $VENV"; else plan "no Python 3.10-3.13 found: would download uv (astral.sh) into $PH/uv-bin and Python 3.12 into $PH/python"; fi
    else
      setup_venv
    fi

    # ---------------------------------------------------------------- 3. dependencies
    step 3 "Installing the app's libraries (a few minutes the first time; downloads roughly 300 MB)"
    if [ "$DRY" = 1 ]; then
      plan "$VPY -m pip install -r $APP/requirements-app.txt -r $APP/install/extra-requirements.txt"
      plan "write $VENV/.../polony_review.pth pointing at $APP/src"
    else
      "$VPY" -m pip install --disable-pip-version-check --upgrade pip >/dev/null 2>&1 || warn "could not upgrade pip (continuing)"
      "$VPY" -m pip install -q --disable-pip-version-check -r "$APP/requirements-app.txt" -r "$APP/install/extra-requirements.txt" || die "installing the libraries failed (see the messages above)"
      SP="$("$VPY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
      printf '%s\n' "$APP/src" > "$SP/polony_review.pth"
    fi

    # ---------------------------------------------------------------- 4. config + checks
    step 4 "Checking the installation"
    if [ "$DRY" = 1 ]; then
      plan "create $DATA, write $PH/config.json, import bigfish/tifffile/fastapi, run 'python -m polony_app.launcher --self-test'"
    else
      mkdir -p "$DATA" "$PH/logs"
      POLONY_HOME="$PH" POLONY_CFG_DATA="$DATA" "$VPY" "$APP/install/write_config.py"
      "$VPY" "$APP/install/check_imports.py" | sed 's/^/  /' || die "the installed libraries cannot be imported"
      say "  starting a temporary server to test it (the first start takes a while) ..."
      POLONY_HOME="$PH" POLONY_CI=1 "$VPY" -m polony_app.launcher --self-test | sed 's/^/  /' || die "the self-test failed"
    fi

  }
  [ "$SHORTCUTS_ONLY" = 1 ] || install_everything

  # ---------------------------------------------------------------- 5. helper commands
  step 5 "Creating the 'Polony Review' icon"
  BINDIR="$PH/bin"
  ICONS="$APP/install/assets"
  APPS="$HOME/Applications"
  if [ "$OS" = linux ]; then
    DESK="$(xdg-user-dir DESKTOP 2>/dev/null || true)"; [ -n "$DESK" ] || DESK="$HOME/Desktop"
  else
    DESK="$HOME/Desktop"
  fi
  LAPPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"

  write_cli() {
    mkdir -p "$BINDIR"
    {
      printf '#!/bin/sh\n# launch (default) | stop | status\nPOLONY_HOME=%s\nexport POLONY_HOME\n' "$(sq "$PH")"
      printf 'exec %s -m polony_app.launcher "$@"\n' "$(sq "$VPY")"
    } > "$BINDIR/polony-review"
    chmod +x "$BINDIR/polony-review"
  }

  make_app_bundle() {  # $1 = destination .app, $2 = display name, $3 = launcher subcommand ("" or stop), $4 = bundle id
    local dest="$1" name="$2" sub="$3" bid="$4" arg=""
    [ -n "$sub" ] && arg=" $sub"
    rm -rf "$dest"
    mkdir -p "$dest/Contents/MacOS" "$dest/Contents/Resources"
    cp "$ICONS/PolonyReview.icns" "$dest/Contents/Resources/PolonyReview.icns"
    cat > "$dest/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>$name</string>
  <key>CFBundleDisplayName</key><string>$name</string>
  <key>CFBundleExecutable</key><string>launcher</string>
  <key>CFBundleIdentifier</key><string>$bid</string>
  <key>CFBundleIconFile</key><string>PolonyReview</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>10.13</string>
  <key>LSUIElement</key><true/>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST
    {
      printf '#!/bin/bash\nPOLONY_HOME=%s\nexport POLONY_HOME\n' "$(sq "$PH")"
      printf 'exec %s -m polony_app.launcher%s "$@"\n' "$(sq "$VPY")" "$arg"
    } > "$dest/Contents/MacOS/launcher"
    chmod +x "$dest/Contents/MacOS/launcher"
    xattr -dr com.apple.quarantine "$dest" >/dev/null 2>&1 || true
  }

  make_linux_desktop() {  # $1 = destination file
    local f="$1"
    cat > "$f" <<DESKTOP
[Desktop Entry]
Type=Application
Version=1.0
Name=Polony Review
Comment=Review and approve polony counts (runs locally in your browser)
Exec="$BINDIR/polony-review"
Icon=$ICONS/PolonyReview.png
Terminal=false
Categories=Science;Education;
Actions=Stop;

[Desktop Action Stop]
Name=Stop Polony Review
Exec="$BINDIR/polony-review" stop
DESKTOP
    chmod +x "$f"
    gio set "$f" metadata::trusted true >/dev/null 2>&1 || true
  }

  if [ "$DRY" = 1 ]; then
    plan "write $BINDIR/polony-review (launch|stop|status)"
    if [ "$OS" = mac ]; then
      plan "create $APPS/Polony Review.app, $APPS/Stop Polony Review.app and a copy $DESK/Polony Review.app"
    else
      plan "create $LAPPS/polony-review.desktop and $DESK/Polony Review.desktop"
    fi
  else
    write_cli
    if [ "$OS" = mac ]; then
      mkdir -p "$APPS" "$DESK"
      make_app_bundle "$APPS/Polony Review.app" "Polony Review" "" "org.polonyreview.app"
      make_app_bundle "$APPS/Stop Polony Review.app" "Stop Polony Review" "stop" "org.polonyreview.stop"
      rm -rf "$DESK/Polony Review.app"; cp -R "$APPS/Polony Review.app" "$DESK/Polony Review.app"
      say "  created $APPS/Polony Review.app and the Desktop icon $DESK/Polony Review.app"
    else
      mkdir -p "$LAPPS" "$DESK"
      make_linux_desktop "$LAPPS/polony-review.desktop"
      make_linux_desktop "$DESK/Polony Review.desktop"
      say "  created $DESK/Polony Review.desktop and an application-menu entry"
    fi
  fi

  # ---------------------------------------------------------------- 6. done
  step 6 "Done"
  say "  Start it from the 'Polony Review' icon on your Desktop (first start takes up to a minute)."
  say "  Stop it:    $BINDIR/polony-review stop"
  say "  Update:     run the same install command again."
  say "  Uninstall:  bash $APP/install/uninstall.sh      (add --purge-data to also delete $DATA)"
  say "  Your results are stored in: $DATA"
  if [ "$DRY" = 0 ] && [ "$NO_LAUNCH" = 0 ]; then
    say ""; say "Starting Polony Review now ..."
    "$BINDIR/polony-review" || warn "could not start now; use the Desktop icon"
  fi
}

main "$@"
