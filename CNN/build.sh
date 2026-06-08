#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
echo "Building CNN (part of NN monorepo)..."
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release} && make -j$(nproc)
