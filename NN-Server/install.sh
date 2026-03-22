#!/bin/bash
set -e

# NN-Server installer
# Downloads from GitHub, builds from source, and installs the binary.
#
# Usage:
#   ./install.sh /usr/local/bin
#   ./install.sh ~/bin
#
# Authentication (private repo):
#   The script tries SSH first (git@github.com). If SSH is not configured,
#   it falls back to HTTPS and prompts for a GitHub Personal Access Token.

INSTALL_DIR="${1:-}"
REPO_SSH="git@github.com:naschpitz/NN-Server.git"
REPO_HTTPS="https://github.com/naschpitz/NN-Server.git"
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
  echo "Missing required dependencies: ${missing[*]}"
  echo ""

  # Detect package manager and build the install command
  install_cmd=""
  if command -v apt &>/dev/null; then
    install_cmd="sudo apt install -y git cmake make g++ qtbase5-dev"
  elif command -v dnf &>/dev/null; then
    install_cmd="sudo dnf install -y git cmake make gcc-c++ qt5-qtbase-devel"
  elif command -v pacman &>/dev/null; then
    install_cmd="sudo pacman -S --noconfirm git cmake make gcc qt5-base"
  elif command -v zypper &>/dev/null; then
    install_cmd="sudo zypper install -y git cmake make gcc-c++ libqt5-qtbase-devel"
  fi

  if [ -n "$install_cmd" ]; then
    echo "The following command will install them:"
    echo "  $install_cmd"
    echo ""
    read -rp "Do you want to install them now? [y/N] " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
      echo "Installing dependencies..."
      $install_cmd
    else
      echo "Aborted. Please install the dependencies manually and re-run this script."
      exit 1
    fi
  else
    echo "Could not detect your package manager."
    echo "Please install the following manually and re-run this script:"
    echo "  git, cmake, make, g++ (or gcc-c++), Qt5/Qt6 development headers"
    exit 1
  fi
fi

echo "  All dependencies found."

# ---------------------------------------------------------------------------
# Clone (private repo — try SSH first, fall back to HTTPS with PAT)
# ---------------------------------------------------------------------------

echo ""
echo "Cloning NN-Server (private repository)..."

cloned=false

# Try SSH first
if ssh -T git@github.com 2>&1 | grep -qi "successfully authenticated"; then
  echo "  SSH authentication detected, cloning via SSH..."
  if git clone --recursive "$REPO_SSH" "$TMP_DIR/NN-Server" 2>/dev/null; then
    cloned=true
  fi
fi

# Fall back to HTTPS with Personal Access Token
if [ "$cloned" = false ]; then
  echo "  SSH authentication not available or failed."
  echo "  To clone this private repository via HTTPS, a GitHub Personal Access Token (PAT) is required."
  echo "  You can create one at: https://github.com/settings/tokens"
  echo "  (select at least the 'repo' scope)"
  echo ""
  read -rp "Enter your GitHub Personal Access Token (or press Enter to abort): " token

  if [ -z "$token" ]; then
    echo "Aborted. Cannot clone without authentication."
    exit 1
  fi

  REPO_PAT="https://${token}@github.com/naschpitz/NN-Server.git"
  git clone --recursive "$REPO_PAT" "$TMP_DIR/NN-Server"
  cloned=true
fi

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
