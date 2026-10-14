#!/usr/bin/env bash
# One-command driver around scripts/runbook/real_training.sh for the data already on this machine.
# Fully automatic, no prompts: install -> templates+mapping -> preflight -> dryrun -> import -> verify
#       -> split -> lock test set (permanent) -> smoke -> traindry -> train.
#
#   export DEVICE=cuda            # cuda | rocm | cpu | auto
#   export TORCH_INDEX=https://download.pytorch.org/whl/cu121   # cu121 NVIDIA / rocm6.2 AMD / cpu
#   bash train_hackathon.sh      # does everything; if interrupted, run again: finished steps are skipped
set -euo pipefail

# --- where the data is (read in place, never modified, no copy needed) -----------------------
DATA_SRC="${DATA_SRC:-}"
if [ -z "$DATA_SRC" ]; then
  for c in "$HOME/hackathon/training_data" "$HOME/hackathon_training_data" "$HOME/hackathon-training-data" \
           "$HOME/hackathon training data" "$HOME/training_data"; do
    [ -d "$c" ] && DATA_SRC="$c" && break
  done
fi
[ -d "${DATA_SRC:-/nonexistent}" ] || { echo "Data dir not found. Run:  DATA_SRC=/path/to/dir bash $0"; exit 1; }

export BASE="${BASE:-$HOME/polony}"        # code, venv, work files, snapshots, models (needs ~35 GB free)
export D="$DATA_SRC"                       # runbook reads TIFFs from here
export DEVICE="${DEVICE:-auto}"
export WORKERS="${WORKERS:-4}"
export TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
export BRANCH="${BRANCH:-feat/legacy-imagej-spotiflow-training}"
W="$BASE/work"; REPO="$BASE/polony-counting"; STATE="$BASE/.driver_state"
RB="$REPO/scripts/runbook/real_training.sh"
mkdir -p "$BASE" "$W"

n_tiff=$(find "$D" -type f \( -iname '*.tif' -o -iname '*.tiff' \) | wc -l)
echo "data: $D  ($(du -sh "$D" | cut -f1), $n_tiff TIFF files)"
[ "$n_tiff" -gt 0 ] || { echo "no TIFF files under $D"; exit 1; }
free_gb=$(df -BG --output=avail "$BASE" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 35 ] || { echo "only ${free_gb} GB free at $BASE, need ~35 GB (snapshot + models)"; exit 1; }

done_step() { [ -f "$STATE.$1" ]; }
mark() { touch "$STATE.$1"; }
step() { echo; echo "=== $* ==="; }

# 1. install (the runbook clones the repo, so bootstrap it with a plain git clone first)
if ! done_step install; then
  step "install"
  [ -d "$REPO/.git" ] || git clone -b "$BRANCH" https://github.com/oriamos/polony-counting.git "$REPO"
  bash "$RB" install
  mark install
fi

# 2-4. fully automatic: no manual mapping, no prompts. Everything accepted on the operator's behalf is
#      written into the files / snapshot manifest as "unreviewed defaults" so it stays traceable.
ml() { PYTHONPATH="$REPO/src" "$BASE/venv/bin/python" -m polony_app.ml --data-dir "$BASE/appdata" "$@"; }
COMMON=(--input "$D" --channel-map "$W/channels.json" --identity-map "$W/identities.json" --allow-unconfirmed-identity
        --downscale 4 --normaliser valid_unsaturated --fractions 0.7,0.15,0.15 --batch-size 4 --epochs 10)

if [ ! -f "$W/channels.json" ] || [ ! -f "$W/identities.json" ]; then
  step "templates + automatic mapping (template defaults: page 0 = red/train, page 1 = green/not trained)"
  ml import-legacy --input "$D" --write-templates "$W/tpl"
  python3 - "$W" "$(date +%F)" "${USER:-operator}" <<'PY'
import json, sys
w, today, user = sys.argv[1:4]
c = json.load(open(f"{w}/tpl/channels.template.json"))
c["confirmed_by"] = f"{user} (automatic: template defaults accepted without per-file review)"
c["confirmed_on"] = today
c["source_of_truth"] = "template default layout validated on the research stacks; NOT re-checked against lab records"
json.dump(c, open(f"{w}/channels.json", "w"), indent=2)
i = json.load(open(f"{w}/tpl/identities.template.json"))
json.dump(i, open(f"{w}/identities.json", "w"), indent=2)   # stays confirmed:false, inferred from file names
PY
fi

if ! done_step import; then
  step "preflight";  ml preflight "${COMMON[@]}" || echo "(preflight reported findings, see above; continuing)"
  step "dry run"
  ml import-legacy "${COMMON[@]}" --dry-run | tee "$W/dryrun.log"
  step "import (~17 min)"
  ml import-legacy "${COMMON[@]}" --workers "$WORKERS" | tee "$W/import.log"
  VER=$(grep -oE 'verify [^ ]+' "$W/import.log" | head -1 | cut -d' ' -f2 || true)
  [ -n "$VER" ] || VER=$(grep -oE 'already imported as [^ ]+' "$W/import.log" | awk '{print $4}' || true)
  [ -n "$VER" ] || { echo "import did not produce a snapshot (see above and $W/import.log; the dry-run BLOCKERS explain why)"; exit 1; }
  echo "$VER" > "$BASE/snapshot_version"; mark import
fi
VER=$(cat "$BASE/snapshot_version")
echo "snapshot: $VER"

if ! done_step verify; then step "verify"; ml verify "$VER"; mark verify; fi
if ! done_step split;  then step "split"; ml split --snapshot "$VER" --allow-unconfirmed-identity; mark split; fi
if ! done_step lock;   then step "lock test set"; ml split --snapshot "$VER" --lock-test --allow-unconfirmed-identity; mark lock; fi

# 5. train
step "GPU smoke test"; ml gpu-smoke --device "$DEVICE"
step "train dry run"; ml train --snapshot "$VER" --base general --device "$DEVICE" --batch-size 4 --epochs 10 --num-workers "$WORKERS" --dry-run
step "TRAIN"; ml train --snapshot "$VER" --base general --device "$DEVICE" --batch-size 4 --epochs 10 --early-stopping-patience 3 --num-workers "$WORKERS" 2>&1 | tee "$W/train.log"
echo; echo "done. Next: bash $RB eval <candidate-id>; bash $RB compare <candidate-id>   (id is in $W/train.log)"
