#!/bin/bash

# Build script for ANN library
# OpenCLWrapper is managed as a Git submodule in extern/

SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR"

# Initialize submodule if not already done
if [ ! -f "extern/OpenCLWrapper/CMakeLists.txt" ]; then
    echo "Initializing Git submodule..."
    git submodule update --init --recursive
fi

# Update submodule to latest remote commit
echo "Updating submodule to latest..."
git submodule update --remote --merge

echo "Building ANN (with OpenCLWrapper)..."
mkdir -p build
cd build
cmake ..
make -j$(nproc)

