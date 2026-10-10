#!/usr/bin/env bash
# Assemble the self-contained probe bundle (deploy root) in $1 (default: ./.vercel-probe-bundle).
# The repo-root pyproject.toml (legacy torch cu118) is NEVER part of the bundle: the bundle
# root contains only app.py, requirements.txt, .python-version, vercel.json and a snapshot
# of polony_app/{__init__,schemas,detection,imaging}. No resources/ and no TIFFs.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="${1:-$HERE/.vercel-probe-bundle}"
rm -rf "$OUT" && mkdir -p "$OUT/polony_app"
cp "$HERE/probe/app.py" "$HERE/probe/requirements.txt" "$HERE/probe/.python-version" "$HERE/probe/vercel.json" "$OUT/"
cp "$REPO/src/polony_app/__init__.py" "$REPO/src/polony_app/schemas.py" "$OUT/polony_app/"
for d in detection imaging; do
  mkdir -p "$OUT/polony_app/$d"
  cp "$REPO/src/polony_app/$d/"*.py "$OUT/polony_app/$d/"
done
find "$OUT" -name '*.tif*' -delete
echo "staged: $OUT"; find "$OUT" -type f | sort
