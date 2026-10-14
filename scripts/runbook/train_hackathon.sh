#!/usr/bin/env bash
# One-command driver around scripts/runbook/real_training.sh for the data already on this machine.
# Runs: install -> templates -> (stops for manual mapping) -> preflight -> dryrun -> import -> verify
#       -> split -> (lock, only if LOCK=yes) -> smoke -> traindry -> train.
#
#   export DEVICE=cuda            # cuda | rocm | cpu | auto
#   export TORCH_INDEX=https://download.pytorch.org/whl/cu121   # cu121 NVIDIA / rocm6.2 AMD / cpu
#   bash train_hackathon.sh                 # run; stops at the first thing that needs a human
#   bash train_hackathon.sh                 # run again after fixing it: finished steps are skipped
#   LOCK=yes bash train_hackathon.sh        # also freeze the test split (PERMANENT) and train
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

# 2. templates, then a human must confirm the channel/identity maps
if [ ! -f "$W/channels.json" ] || [ ! -f "$W/identities.json" ]; then
  [ -d "$W/tpl" ] || { step "templates"; bash "$RB" templates; }
  cat <<MSG

STOP: manual step. These files encode lab knowledge and must be confirmed by a person:
  cp $W/tpl/channels.template.json   $W/channels.json     # fill confirmed_by, confirmed_on, source_of_truth
  cp $W/tpl/identities.template.json $W/identities.json   # check every row against lab records, set "confirmed": true
Details: $REPO/docs/app/LEGACY_FINETUNING.md section 3.  Then run this script again.
MSG
  exit 0
fi

# 3. checks, import
if ! done_step import; then
  step "preflight";  bash "$RB" preflight
  step "dry run (BLOCKERS must be empty)"
  bash "$RB" dryrun | tee "$W/dryrun.log"
  if grep -qiE 'blocker' "$W/dryrun.log" && ! grep -qiE 'blockers?: *(none|0)' "$W/dryrun.log"; then
    echo; echo "STOP: the dry run lists BLOCKERS (see $W/dryrun.log). Fix them, then run again."; exit 1
  fi
  step "import (~17 min)"
  bash "$RB" import | tee "$W/import.log"
  VER=$(grep -oE 'verify [^ ]+' "$W/import.log" | head -1 | cut -d' ' -f2 || true)
  [ -n "$VER" ] || VER=$(grep -oE 'already imported as [^ ]+' "$W/import.log" | awk '{print $4}' || true)
  [ -n "$VER" ] || { echo "could not read the snapshot version from $W/import.log"; exit 1; }
  echo "$VER" > "$BASE/snapshot_version"; mark import
fi
VER=$(cat "$BASE/snapshot_version")
echo "snapshot: $VER"

if ! done_step verify; then step "verify"; bash "$RB" verify "$VER"; mark verify; fi
if ! done_step split;  then step "split (inspect the group table)"; bash "$RB" split "$VER"; mark split; fi

# 4. lock is permanent: only on explicit request
if ! done_step lock; then
  if [ "${LOCK:-no}" = yes ]; then
    step "lock test set (PERMANENT)"; bash "$RB" lock "$VER"; mark lock
  else
    cat <<MSG

STOP: the test split is not frozen yet. Look at the split table above; if it is right, run:
  LOCK=yes bash $0
(freezing the test groups is permanent, so it is never done automatically)
MSG
    exit 0
  fi
fi

# 5. train
step "GPU smoke test"; bash "$RB" smoke
step "train dry run"; bash "$RB" traindry "$VER"
step "TRAIN"; bash "$RB" train "$VER" 2>&1 | tee "$W/train.log"
echo; echo "done. Next: bash $RB eval <candidate-id>; bash $RB compare <candidate-id>   (id is in $W/train.log)"
