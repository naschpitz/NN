#!/bin/bash

# Build script for ANN library

SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR"

OPENCLWRAPPER_DIR="$SCRIPT_DIR/../OpenCLWrapper"
LIBS_OPENCLWRAPPER_DIR="$SCRIPT_DIR/libs/OpenCLWrapper"

# Copy the latest OpenCLWrapper library and headers
echo "Copying OpenCLWrapper library and headers..."
cp "$OPENCLWRAPPER_DIR/build/libOpenCLWrapper.a" "$LIBS_OPENCLWRAPPER_DIR/"
cp "$OPENCLWRAPPER_DIR"/*.hpp "$LIBS_OPENCLWRAPPER_DIR/"

echo "Building ANN..."
mkdir -p build
cd build
cmake ..
make -j$(nproc)

