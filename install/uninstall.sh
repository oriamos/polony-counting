#!/usr/bin/env bash
# Polony Review uninstaller (macOS/Linux). Removes the app, its Python environment and the shortcuts.
# Your data folder (default ~/PolonyReviewData) and downloaded AI models (~/PolonyReview/models) are KEPT unless --purge-data.
#   bash ~/PolonyReview/app/install/uninstall.sh [--purge-data] [--dry-run]
set -euo pipefail

main() {
  PH="${POLONY_HOME:-$HOME/PolonyReview}"
  PURGE=0; DRY=0
  for a in "$@"; do
    case "$a" in
      --purge-data) PURGE=1 ;;
      --dry-run) DRY=1 ;;
      -h|--help) sed -n '2,5p' "$0"; return 0 ;;
      *) echo "Unknown option: $a" >&2; return 2 ;;
    esac
  done
  case "$(uname -s)" in Darwin) OS=mac ;; *) OS=linux ;; esac
  DATA="${POLONY_DATA_DIR:-}"
  if [ -z "$DATA" ] && [ -f "$PH/config.json" ] && [ -x "$PH/venv/bin/python" ]; then
    DATA="$("$PH/venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("data_dir",""))' "$PH/config.json" 2>/dev/null || true)"
  fi
  [ -n "$DATA" ] || DATA="$HOME/PolonyReviewData"
  if [ "$OS" = linux ]; then
    DESK="$(xdg-user-dir DESKTOP 2>/dev/null || true)"; [ -n "$DESK" ] || DESK="$HOME/Desktop"
  else
    DESK="$HOME/Desktop"
  fi
  LAPPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"

  rm_path() {
    if [ -e "$1" ] || [ -L "$1" ]; then
      if [ "$DRY" = 1 ]; then echo "  [dry-run] would remove $1"; else rm -rf "$1" && echo "  removed $1"; fi
    fi
  }

  echo "Uninstalling Polony Review from $PH"
  if [ "$DRY" = 0 ] && [ -x "$PH/venv/bin/python" ]; then
    POLONY_HOME="$PH" "$PH/venv/bin/python" -m polony_app.launcher stop || true
  elif [ "$DRY" = 1 ]; then
    echo "  [dry-run] would stop a running server"
  fi
  rm_path "$HOME/Applications/Polony Review.app"
  rm_path "$HOME/Applications/Stop Polony Review.app"
  rm_path "$DESK/Polony Review.app"
  rm_path "$DESK/Polony Review.desktop"
  rm_path "$LAPPS/polony-review.desktop"
  # the install folder (app, venv, downloaded Python, uv, logs, config)
  case "$PH" in ""|"/"|"$HOME") echo "Refusing to remove '$PH'" >&2; return 1 ;; esac
  if [ "$PURGE" = 0 ] && [ -d "$PH/models" ]; then
    # keep the downloaded AI models (~90 MB each) unless --purge-data; everything else goes
    for e in "$PH"/* "$PH"/.[!.]*; do
      [ -e "$e" ] || [ -L "$e" ] || continue
      [ "$e" = "$PH/models" ] && continue
      rm_path "$e"
    done
    echo "  kept the downloaded AI models in $PH/models (use --purge-data to delete them too)"
  else
    rm_path "$PH"
  fi
  if [ "$PURGE" = 1 ]; then
    case "$DATA" in ""|"/"|"$HOME") echo "Refusing to remove '$DATA'" >&2; return 1 ;; esac
    rm_path "$DATA"
  elif [ -d "$DATA" ]; then
    echo "  kept your data in $DATA (delete it yourself, or re-run with --purge-data)"
  fi
  echo "Done."
}

main "$@"
