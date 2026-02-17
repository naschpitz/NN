#!/bin/bash

# Build script for ANN library
# OpenCLWrapper is expected to be at ../OpenCLWrapper (sibling directory)
# It will be built automatically via CMake add_subdirectory

SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR"

# Check that OpenCLWrapper exists
if [ ! -f "../OpenCLWrapper/CMakeLists.txt" ]; then
    echo "Error: OpenCLWrapper not found at ../OpenCLWrapper"
    echo "Please ensure OpenCLWrapper repository is cloned as a sibling directory."
    exit 1
fi

echo "Building ANN (with OpenCLWrapper)..."
mkdir -p build
cd build
cmake ..
make -j$(nproc)

