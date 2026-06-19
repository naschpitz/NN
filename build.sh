#!/usr/bin/env bash
#
# Build every NN component into its OWN build directory (not a single root build).
#
# Two build profiles keep dev iteration from clobbering a release executable that
# may be mid-training in the other directory:
#   --development  (default)  ->  <component>/build-dev   (CMAKE_BUILD_TYPE=Debug)
#   --release                 ->  <component>/build       (CMAKE_BUILD_TYPE=Release)
#
# Qt6 kit:
#   --static       (default)
#   --shared
#
# Usage:
#   ./build.sh                       # dev (Debug), static Qt6 — the default
#   ./build.sh --release             # release (Release), static Qt6
#   ./build.sh --development --shared
#   ./build.sh --release --shared
#   ./build.sh --help
#
# Each component is configured & built standalone — it pulls its dependencies via
# add_subdirectory (CNN->ANN->extern/OpenCLWrapper, etc.) — so its artifacts live
# under <component>/<build-dir>. The Qt6 kit paths are kept in ONE place, this
# repo's root CMakeUserPresets.json (git-ignored, machine-specific); this script
# copies that master into each component, then configures and builds it.
#
# The preset only pins the Qt6 kit (CMAKE_PREFIX_PATH); this script overrides the
# preset's binaryDir/buildType with -B / -DCMAKE_BUILD_TYPE (verified on CMake >= 3.21).
#
# Requires a root CMakeUserPresets.json pinning your Qt6 kit (see README).

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="development"
KIT="static"
for arg in "$@"; do
  case "$arg" in
    --development) MODE="development" ;;
    --release)     MODE="release" ;;
    --static)      KIT="static" ;;
    --shared)      KIT="shared" ;;
    -h|--help)
      sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "error: unknown argument '$arg'" >&2
      echo "       expected --development|--release|--static|--shared" >&2
      exit 1
      ;;
  esac
done

if [[ "$MODE" == "development" ]]; then
  BUILD_DIR="build-dev"
  BUILD_TYPE="Debug"
else
  BUILD_DIR="build"
  BUILD_TYPE="Release"
fi

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
  echo "  $c   (kit: $KIT, mode: $MODE, dir: $BUILD_DIR, type: $BUILD_TYPE)"
  echo "================================================================"
  # Distribute the master preset; ${sourceDir} in it re-resolves to this component.
  cp "$MASTER" "$ROOT/$c/CMakeUserPresets.json"
  if ( cd "$ROOT/$c" && cmake --preset "$KIT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" && cmake --build "$BUILD_DIR" -j"$jobs" ); then
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
echo "All components built into their own $BUILD_DIR/ dirs: mode=$MODE, kit=$KIT, type=$BUILD_TYPE"
