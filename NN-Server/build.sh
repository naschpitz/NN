#!/bin/bash
#
# Build NN-Server standalone.
#
# Usage:
#   ./build.sh               # development (Debug) into build-dev — default
#   ./build.sh --release     # release (Release) into build
#   ./build.sh --development
#
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODE="development"
for arg in "$@"; do
  case "$arg" in
    --development) MODE="development" ;;
    --release)     MODE="release" ;;
    -h|--help)     echo "usage: $0 [--development|--release]"; echo "  --development  (default) Debug into build-dev"; echo "  --release               Release into build"; exit 0 ;;
    *) echo "usage: $0 [--development|--release] (default --development)" >&2; exit 1 ;;
  esac
done

if [[ "$MODE" == "development" ]]; then
  BUILD_DIR="build-dev"; BUILD_TYPE="Debug"
else
  BUILD_DIR="build";     BUILD_TYPE="Release"
fi

echo "Building NN-Server (part of NN monorepo) into $BUILD_DIR ($BUILD_TYPE)..."
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE" && make -j$(nproc)
