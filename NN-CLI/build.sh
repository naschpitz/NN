#!/bin/bash

# Build script for ANN-CLI
# Dependencies (OpenCLWrapper and ANN) are managed as Git submodules in extern/

SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR"

# Initialize submodules if not already done
if [ ! -f "extern/ANN/CMakeLists.txt" ] || [ ! -f "extern/OpenCLWrapper/CMakeLists.txt" ]; then
    echo "Initializing Git submodules..."
    git submodule update --init --recursive
fi

echo "Building ANN-CLI (with dependencies)..."
mkdir -p build
cd build
cmake ..
make -j$(nproc)

