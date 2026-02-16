#!/bin/bash

# Build script for ANN-CLI

SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR"

ANN_DIR="$SCRIPT_DIR/../ANN"
LIBS_ANN_DIR="$SCRIPT_DIR/libs/ANN"
LIBS_NLOHMANN_DIR="$SCRIPT_DIR/libs/nlohmann"
LIBS_OPENCL_WRAPPER_DIR="$SCRIPT_DIR/libs/OpenCLWrapper"

# Copy the latest ANN library and headers
echo "Copying ANN library and headers..."
mkdir -p "$LIBS_ANN_DIR"
cp "$ANN_DIR/build/libANN.a" "$LIBS_ANN_DIR/"
cp "$ANN_DIR"/*.hpp "$LIBS_ANN_DIR/"

# Copy the nlohmann JSON library
echo "Copying nlohmann JSON library..."
mkdir -p "$LIBS_NLOHMANN_DIR"
cp -r "$ANN_DIR/libs/nlohmann/"* "$LIBS_NLOHMANN_DIR/"

# Copy the OpenCLWrapper library
echo "Copying OpenCLWrapper library..."
mkdir -p "$LIBS_OPENCL_WRAPPER_DIR"
cp "$ANN_DIR/libs/OpenCLWrapper/libOpenCLWrapper.a" "$LIBS_OPENCL_WRAPPER_DIR/"
cp "$ANN_DIR/libs/OpenCLWrapper/"*.hpp "$LIBS_OPENCL_WRAPPER_DIR/"

echo "Building ANN-CLI..."
mkdir -p build
cd build
cmake ..
make -j$(nproc)

