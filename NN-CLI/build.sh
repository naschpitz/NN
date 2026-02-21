#!/bin/bash

# Build script for NN-CLI
# Dependencies (OpenCLWrapper, ANN, and CNN) are managed as Git submodules in extern/

SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR"

# Initialize submodules if not already done
if [ ! -f "extern/CNN/CMakeLists.txt" ] || [ ! -f "extern/ANN/CMakeLists.txt" ] || [ ! -f "extern/OpenCLWrapper/CMakeLists.txt" ]; then
    echo "Initializing Git submodules..."
    git submodule update --init --recursive
fi

# Update submodules to latest remote commits
echo "Updating submodules to latest..."
git submodule update --remote --merge

echo "Building NN-CLI (with dependencies)..."
mkdir -p build
cd build
cmake ..
make -j$(nproc)

