#!/usr/bin/env bash
#
# Build every NN component into its OWN build/ directory (not a single root build).
#
# Each component is configured & built standalone — it pulls its dependencies via
# add_subdirectory (CNN→ANN→extern/OpenCLWrapper, etc.) — so its artifacts live under
# <component>/build/. The Qt6 kit paths are kept in ONE place, this repo's root
# CMakeUserPresets.json (git-ignored, machine-specific); this script copies that master
# into each component as its CMakeUserPresets.json, then configures and builds it.
#
# Usage:
#   ./build.sh [preset]      preset = static (default) | shared
#
# Requires a root CMakeUserPresets.json pinning your Qt6 kit (see README).

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESET="${1:-static}"
COMPONENTS=(ANN CNN NN-CLI NN-Server)

MASTER="$ROOT/CMakeUserPresets.json"
if [[ ! -f "$MASTER" ]]; then
  echo "error: $MASTER not found." >&2
  echo "       Create it (git-ignored) with your Qt6 kit paths — see README.md." >&2
  exit 1
fi

# Ensure the OpenCLWrapper submodules under each extern/ are present.
git -C "$ROOT" submodule update --init --recursive

jobs="$(nproc 2>/dev/null || echo 4)"
failed=()

for c in "${COMPONENTS[@]}"; do
  echo "================================================================"
  echo "  $c   (preset: $PRESET)"
  echo "================================================================"
  # Distribute the master preset; ${sourceDir} in it re-resolves to this component,
  # so binaryDir becomes <component>/build.
  cp "$MASTER" "$ROOT/$c/CMakeUserPresets.json"
  if ( cd "$ROOT/$c" && cmake --preset "$PRESET" && cmake --build build -j"$jobs" ); then
    echo "==> $c: OK"
  else
    echo "==> $c: FAILED" >&2
    failed+=("$c")
  fi
done

echo
if (( ${#failed[@]} )); then
  echo "Build finished with failures: ${failed[*]}" >&2
  exit 1
fi
echo "All components built into their own build/ dirs: ${COMPONENTS[*]}"
