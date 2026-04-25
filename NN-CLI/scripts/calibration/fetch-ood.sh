#!/usr/bin/env bash
#
# Fetch the OOD calibration dataset (DTD textures + Places365 val_256
# scenes) and generate a synthetic OOD set. Idempotent — skips
# components that are already on disk.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OOD_DIR="$REPO_ROOT/extern-datasets/ood"

mkdir -p "$OOD_DIR"

# ----- DTD ------------------------------------------------------------------
DTD_DIR="$OOD_DIR/dtd"
DTD_TGZ="$OOD_DIR/dtd-r1.0.1.tar.gz"
DTD_URL="https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"

if [ -d "$DTD_DIR/images" ] && [ "$(find "$DTD_DIR/images" -type f -name '*.jpg' | head -1)" ]; then
  echo "[skip] DTD already extracted at $DTD_DIR/images"
else
  echo "[fetch] DTD (~600 MB) → $DTD_DIR"
  mkdir -p "$DTD_DIR"
  curl -L --fail -o "$DTD_TGZ" "$DTD_URL"
  tar -xzf "$DTD_TGZ" -C "$DTD_DIR" --strip-components=1
  rm "$DTD_TGZ"
fi

# ----- Places365 (val_256, ~500MB compressed, 36 500 images) ----------------
PL_DIR="$OOD_DIR/places365-val"
PL_TAR="$OOD_DIR/val_256.tar"
PL_URL="http://data.csail.mit.edu/places/places365/val_256.tar"

if [ -d "$PL_DIR/val_256" ] && [ "$(find "$PL_DIR/val_256" -type f -name '*.jpg' | head -1)" ]; then
  echo "[skip] Places365 val_256 already extracted at $PL_DIR/val_256"
else
  echo "[fetch] Places365 val_256 (~2 GB) → $PL_DIR"
  mkdir -p "$PL_DIR"
  curl -L --fail -o "$PL_TAR" "$PL_URL"
  tar -xf "$PL_TAR" -C "$PL_DIR"
  rm "$PL_TAR"
fi

# ----- Synthetic ------------------------------------------------------------
SYN_DIR="$OOD_DIR/synthetic"
echo "[gen]   Synthetic OOD → $SYN_DIR"
python3 "$SCRIPT_DIR/make-synthetic-ood.py" "$SYN_DIR"

# ----- Summary --------------------------------------------------------------
echo
echo "OOD set ready at $OOD_DIR:"
for sub in dtd places365-val synthetic; do
  if [ -d "$OOD_DIR/$sub" ]; then
    n=$(find "$OOD_DIR/$sub" -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l)
    printf "  %-20s %s images\n" "$sub" "$n"
  fi
done
