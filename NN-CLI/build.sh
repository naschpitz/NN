#!/bin/bash

# Build script for ANN-CLI

SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR"

ANN_DIR="$SCRIPT_DIR/../ANN"
LIBS_ANN_DIR="$SCRIPT_DIR/libs/ANN"

# Copy the latest ANN library and headers
echo "Copying ANN library and headers..."
cp "$ANN_DIR/build/libANN.a" "$LIBS_ANN_DIR/"
cp "$ANN_DIR"/*.hpp "$LIBS_ANN_DIR/"

echo "Building ANN-CLI..."
mkdir -p build
cd build
cmake ..
make -j$(nproc)

