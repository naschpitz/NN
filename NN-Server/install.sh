#!/bin/bash
set -e

# NN-Server installer
# Downloads from GitHub, builds from source, and installs the binary.
#
# Usage:
#   ./install.sh /usr/local/bin
#   ./install.sh ~/bin
#   curl -sL https://raw.githubusercontent.com/naschpitz/NN-Server/master/install.sh | bash -s -- /usr/local/bin

INSTALL_DIR="${1:-}"
REPO_URL="https://github.com/naschpitz/NN-Server.git"
TMP_DIR="$(mktemp -d)"

cleanup() {
  echo "Cleaning up..."
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Validate arguments
# ---------------------------------------------------------------------------

if [ -z "$INSTALL_DIR" ]; then
  echo "Usage: $0 <install-path>"
  echo ""
  echo "  <install-path>  Directory where the NN-Server binary will be installed."
  echo ""
  echo "Examples:"
  echo "  $0 /usr/local/bin"
  echo "  $0 \$HOME/bin"
  exit 1
fi

# Resolve to absolute path
INSTALL_DIR="$(mkdir -p "$INSTALL_DIR" && cd "$INSTALL_DIR" && pwd)"

# ---------------------------------------------------------------------------
# Check dependencies
# ---------------------------------------------------------------------------

echo "Checking dependencies..."

missing=()
for cmd in git cmake make g++; do
  if ! command -v "$cmd" &>/dev/null; then
    missing+=("$cmd")
  fi
done

# Check for Qt (qmake or qmake6)
if ! command -v qmake &>/dev/null && ! command -v qmake6 &>/dev/null; then
  missing+=("Qt (qmake)")
fi

if [ ${#missing[@]} -gt 0 ]; then
  echo "ERROR: Missing required dependencies: ${missing[*]}"
  echo ""
  echo "Install them with your package manager, e.g.:"
  echo "  sudo apt install git cmake make g++ qtbase5-dev"
  echo "  sudo dnf install git cmake make gcc-c++ qt5-qtbase-devel"
  exit 1
fi

echo "  All dependencies found."

# ---------------------------------------------------------------------------
# Clone
# ---------------------------------------------------------------------------

echo ""
echo "Cloning NN-Server..."
git clone --recursive "$REPO_URL" "$TMP_DIR/NN-Server"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

echo ""
echo "Building NN-Server..."
cd "$TMP_DIR/NN-Server"
mkdir -p build
cd build
cmake ..
make -j"$(nproc)"

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

echo ""
echo "Installing NN-Server to $INSTALL_DIR ..."

if [ -w "$INSTALL_DIR" ]; then
  cp NN-Server "$INSTALL_DIR/NN-Server"
else
  echo "  (requires sudo)"
  sudo cp NN-Server "$INSTALL_DIR/NN-Server"
fi

chmod +x "$INSTALL_DIR/NN-Server"

echo ""
echo "========================================"
echo "  NN-Server installed successfully!"
echo "  Binary: $INSTALL_DIR/NN-Server"
echo "========================================"
echo ""
echo "Quick start:"
echo "  export NN_MODEL_CONFIG=/path/to/model.json"
echo "  $INSTALL_DIR/NN-Server"
echo ""
echo "See documentation: https://naschpitz.github.io/NN-Server/"
