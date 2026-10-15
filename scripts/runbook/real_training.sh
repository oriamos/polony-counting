#!/usr/bin/env bash
# Step-by-step runbook for fine-tuning Spotiflow on the ~1,500 historical ImageJ TIFFs.
# Each step is a sub-command; nothing runs the next step for you, and nothing promotes a model.
#
#   export BASE=/data/polony          # one big disk: >= 60 GB free (input ~26 GB, snapshot 4-9 GB, models)
#   bash scripts/runbook/real_training.sh install
#   ... see "usage" below for the order.
set -euo pipefail

BASE="${BASE:?set BASE to a directory on a disk with >= 60 GB free}"
REPO="${REPO:-$BASE/polony-counting}"
BRANCH="${BRANCH:-feat/legacy-imagej-spotiflow-training}"
VENV="${VENV:-$BASE/venv}"
D="${D:-$BASE/tiffs}"            # historical TIFFs (copied here from your storage)
W="${W:-$BASE/work}"             # templates, mapping files, reports
DATA="${DATA:-$BASE/appdata}"    # snapshots (datasets/) and models (models/)
DEVICE="${DEVICE:-auto}"         # auto | rocm | cuda | cpu
WORKERS="${WORKERS:-4}"
# Torch wheel index for the GPU: AMD -> rocm (pick the version matching your driver),
# NVIDIA -> cu121/cu118, none -> cpu.  Check https://pytorch.org/get-started/locally/
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/rocm6.2}"

ml() { PYTHONPATH="$REPO/src" "$VENV/bin/python" -m polony_app.ml --data-dir "$DATA" "$@"; }

case "${1:-usage}" in
install)   # 1. code + python environment (needs Python >= 3.10, git)
  mkdir -p "$BASE" "$W" "$DATA"
  [ -d "$REPO/.git" ] || git clone https://github.com/oriamos/polony-counting.git "$REPO"
  git -C "$REPO" fetch origin "$BRANCH" && git -C "$REPO" checkout "$BRANCH" && git -C "$REPO" pull --ff-only origin "$BRANCH"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  # torch FIRST from the GPU index (do not let spotiflow pull a CPU build over it)
  "$VENV/bin/pip" install torch torchvision --index-url "$TORCH_INDEX"
  "$VENV/bin/pip" install -r "$REPO/requirements-app.txt"
  "$VENV/bin/pip" install -r "$REPO/install/extra-ai-requirements.txt"
  echo "installed; commit: $(git -C "$REPO" rev-parse --short HEAD)  (snapshots depend on the code: do not change it between import and train)"
  ;;
upload)    # 2. get the TIFFs onto THIS machine's disk (the importer reads a local directory)
  cat <<MSG
Copy the TIFFs into: $D   (keep ONE directory tree; the original file names matter for the identity map)
  from your computer :  rsync -ah --info=progress2 /path/to/tiffs/ USER@HOST:$D/
  from S3            :  aws s3 sync s3://BUCKET/PREFIX $D/
  from GCS           :  gsutil -m rsync -r gs://BUCKET/PREFIX $D/
  from any remote    :  rclone copy REMOTE:PATH $D --progress --transfers 8
Then count:  find $D -type f \( -iname '*.tif' -o -iname '*.tiff' \) | wc -l
Note: these are research images; upload only to storage you are permitted to use, never into the git repo.
MSG
  ;;
templates) ml import-legacy --input "$D" --write-templates "$W/tpl" ;;   # 3. then EDIT and confirm (see below)
mapping)
  cat <<MSG
Copy $W/tpl/channels.template.json   -> $W/channels.json   (fill confirmed_by, confirmed_on, source_of_truth)
Copy $W/tpl/identities.template.json -> $W/identities.json  (check each row with the lab records, set "confirmed": true)
Optional: $W/tpl/examined-fields.template.json -> $W/examined-fields.json (verified zeros only)
Details: docs/app/LEGACY_FINETUNING.md section 3.
MSG
  ;;
preflight) ml preflight --input "$D" --channel-map "$W/channels.json" --identity-map "$W/identities.json" ;;     # 4
dryrun)    ml import-legacy --input "$D" --channel-map "$W/channels.json" --identity-map "$W/identities.json" \
             --downscale 4 --normaliser valid_unsaturated --fractions 0.7,0.15,0.15 --batch-size 4 --epochs 10 --dry-run ;;  # 5: BLOCKERS must be empty
import)    ml import-legacy --input "$D" --channel-map "$W/channels.json" --identity-map "$W/identities.json" \
             --downscale 4 --normaliser valid_unsaturated --fractions 0.7,0.15,0.15 --workers "$WORKERS" ;;     # 6 (~17 min)
verify)    ml verify "${2:?snapshot version printed by import}" ;;
split)     ml split --snapshot "${2:?version}" ;;                                   # inspect the group table first
lock)      ml split --snapshot "${2:?version}" --lock-test ;;                       # PERMANENT
smoke)     ml gpu-smoke --device "$DEVICE" ;;
traindry)  ml train --snapshot "${2:?version}" --base general --device "$DEVICE" --batch-size 4 --epochs 10 --num-workers "$WORKERS" --dry-run ;;
bench)     ml train --snapshot "${2:?version}" --base general --device "$DEVICE" --batch-size 4 --epochs 1 --num-workers "$WORKERS" ;;   # note seconds/epoch
train)     ml train --snapshot "${2:?version}" --base general --device "$DEVICE" --batch-size 4 --epochs 10 --early-stopping-patience 3 --num-workers "$WORKERS" ;;
eval)      ml eval-model --model "${2:?candidate id}" --device "$DEVICE"
           ml eval-model --model general --downscale 4 --device "$DEVICE" ;;
compare)   ml compare --models "${2:?candidate id}" general ;;                      # promotion stays a manual decision: ml promote ...
*)
  sed -n 1,12p "$0"
  echo "order: install upload templates mapping preflight dryrun import verify|split|lock <ver> smoke traindry bench train eval compare <cand-id>"
  ;;
esac
