#!/usr/bin/env bash
#
# Calibrate the free-energy OOD threshold for a trained NN-CLI model.
#
# Required:
#   --model FILE          Trained model JSON (e.g. .../output/best_model.json)
#   --id-images DIR       Directory of in-distribution images (recursed)
#   --output-dir DIR      Where to write threshold.json + per-set predict JSONs
#
# Optional:
#   --id-sample-count N   Random subsample size for ID set (default 500)
#   --ood-root DIR        OOD root (default <repo>/extern-datasets/ood)
#   --ood-sample-count N  Random subsample size for OOD set (default 1500)
#   --id-percentile P     ID percentile used as the threshold (default 95)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL=""
ID_IMAGES=""
OUTPUT_DIR=""
ID_SAMPLE_COUNT=500
OOD_SAMPLE_COUNT=1500
OOD_ROOT="$REPO_ROOT/extern-datasets/ood"
ID_PERCENTILE=95

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --id-images) ID_IMAGES="$2"; shift 2;;
    --id-sample-count) ID_SAMPLE_COUNT="$2"; shift 2;;
    --ood-root) OOD_ROOT="$2"; shift 2;;
    --ood-sample-count) OOD_SAMPLE_COUNT="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --id-percentile) ID_PERCENTILE="$2"; shift 2;;
    -h|--help)
      sed -n '/^# /,/^$/p' "$0" | head -25; exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 1;;
  esac
done

for v in MODEL ID_IMAGES OUTPUT_DIR; do
  if [ -z "${!v}" ]; then
    echo "Missing required --${v,,} (lowercased: ${v,,/_/-})" >&2
    exit 1
  fi
done

NN_CLI="$REPO_ROOT/build/NN-CLI"
if [ ! -x "$NN_CLI" ]; then
  echo "NN-CLI binary not found at $NN_CLI — build first (./build.sh)." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ----- Build inputs.json files (random subsamples) -------------------------
sample_inputs() {
  local root="$1" count="$2" out="$3"
  python3 - "$root" "$count" "$out" <<'PY'
import json, os, random, sys
from pathlib import Path
# Absolute paths required: NN-CLI resolves relative entries against the
# input JSON's own directory, not the current working directory.
root = Path(sys.argv[1]).resolve()
count = int(sys.argv[2])
out = Path(sys.argv[3])
random.seed(42)
exts = (".jpg", ".jpeg", ".png", ".bmp")
imgs = [str(p) for p in root.rglob("*") if p.suffix.lower() in exts]
random.shuffle(imgs)
imgs = imgs[:count]
out.write_text(json.dumps({"inputs": imgs}, indent=2))
print(f"  {out.name}: sampled {len(imgs)} images from {root}")
PY
}

echo "[1/3] Sampling ID and OOD inputs..."
sample_inputs "$ID_IMAGES" "$ID_SAMPLE_COUNT" "$OUTPUT_DIR/calibration-id-inputs.json"
sample_inputs "$OOD_ROOT" "$OOD_SAMPLE_COUNT" "$OUTPUT_DIR/calibration-ood-inputs.json"

# ----- Run predict in batch ------------------------------------------------
echo "[2/3] Running NN-CLI predict on ID set..."
"$NN_CLI" --config "$MODEL" --mode predict --input-type image \
  --input "$OUTPUT_DIR/calibration-id-inputs.json" \
  --output "$OUTPUT_DIR/calibration-id-predict.json" \
  --log-level error

echo "      Running NN-CLI predict on OOD set..."
"$NN_CLI" --config "$MODEL" --mode predict --input-type image \
  --input "$OUTPUT_DIR/calibration-ood-inputs.json" \
  --output "$OUTPUT_DIR/calibration-ood-predict.json" \
  --log-level error

# ----- Compute threshold ---------------------------------------------------
echo "[3/3] Computing free-energy distributions and threshold..."
python3 "$SCRIPT_DIR/compute-threshold.py" \
  --id-predict "$OUTPUT_DIR/calibration-id-predict.json" \
  --ood-predict "$OUTPUT_DIR/calibration-ood-predict.json" \
  --id-percentile "$ID_PERCENTILE" \
  --out "$OUTPUT_DIR/threshold.json"

echo
echo "Done. Threshold written to $OUTPUT_DIR/threshold.json"
