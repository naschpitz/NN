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

REPO_DIR="$INSTALL_DIR/NN-Server-repo"

cloned=false

# Check if already cloned (for re-installs / updates)
if [ -d "$REPO_DIR/.git" ]; then
  echo "  Existing installation found at $REPO_DIR"
  echo "  Pulling latest changes..."
  cd "$REPO_DIR"
  git pull
  git submodule update --init --recursive
  cloned=true
fi

if [ "$cloned" = false ]; then
  # Try SSH first
  if ssh -T git@github.com 2>&1 | grep -qi "successfully authenticated"; then
    echo "  SSH authentication detected, cloning via SSH..."
    if git clone --recursive "$REPO_SSH" "$REPO_DIR" 2>/dev/null; then
      cloned=true
    fi
  fi

  # Fall back to HTTPS with Personal Access Token
  if [ "$cloned" = false ]; then
    TOKEN_FILE="$HOME/.nn-server-github-token"
    token=""

    # Try to load a saved token
    if [ -f "$TOKEN_FILE" ]; then
      token=$(cat "$TOKEN_FILE")
      echo "  Using saved GitHub token from $TOKEN_FILE"
    fi

    # If no saved token, ask the user
    if [ -z "$token" ]; then
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

      # Save the token for future use
      echo "$token" > "$TOKEN_FILE"
      chmod 600 "$TOKEN_FILE"
      echo "  Token saved to $TOKEN_FILE"
    fi

    REPO_PAT="https://${token}@github.com/naschpitz/NN-Server.git"
    git clone --recursive "$REPO_PAT" "$REPO_DIR"
    cloned=true
  fi
fi

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

echo ""
echo "Building NN-Server..."
cd "$REPO_DIR"
mkdir -p build
cd build
cmake ..
make -j"$(nproc)"

# ---------------------------------------------------------------------------
# Symlink
# ---------------------------------------------------------------------------

echo ""
echo "Creating symlink..."

# Create a symlink in INSTALL_DIR pointing to the binary inside the repo
ln -sf "$REPO_DIR/build/NN-Server" "$INSTALL_DIR/NN-Server"

echo ""
echo "========================================"
echo "  NN-Server installed successfully!"
echo "  Binary:  $REPO_DIR/build/NN-Server"
echo "  Symlink: $INSTALL_DIR/NN-Server"
echo "========================================"
echo ""
echo "IMPORTANT: Do not delete $REPO_DIR — it contains"
echo "  OpenCL kernel files required at runtime for GPU execution."

# ---------------------------------------------------------------------------
# Systemd service setup
# ---------------------------------------------------------------------------

SERVICE_FILE="/etc/systemd/system/nn-server.service"
BINARY_PATH="$INSTALL_DIR/NN-Server"

# If the service already exists, just restart it
if [ -f "$SERVICE_FILE" ]; then
  echo ""
  echo "Existing systemd service found. Restarting..."
  sudo systemctl daemon-reload
  sudo systemctl restart nn-server
  echo "  nn-server service restarted."
else
  echo ""
  read -rp "Do you want to create a systemd service so NN-Server starts on boot? [y/N] " setup_service

  if [[ "$setup_service" =~ ^[Yy]$ ]]; then
    # Ask for config file path
    DEFAULT_CONFIG="$INSTALL_DIR/config.json"
    read -rp "Path to config.json [$DEFAULT_CONFIG]: " config_path
    config_path="${config_path:-$DEFAULT_CONFIG}"

    # Resolve to absolute path
    if [[ "$config_path" != /* ]]; then
      config_path="$(cd "$(dirname "$config_path")" && pwd)/$(basename "$config_path")"
    fi

    sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=NN-Server Neural Network Inference Server
After=network.target

[Service]
ExecStart=$BINARY_PATH $config_path
WorkingDirectory=$INSTALL_DIR
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable nn-server

    if [ -f "$config_path" ]; then
      sudo systemctl start nn-server
      echo ""
      echo "  nn-server service created, enabled, and started."
    else
      echo ""
      echo "  nn-server service created and enabled."
      echo "  It will start automatically on boot once $config_path exists."
      echo "  To start manually: sudo systemctl start nn-server"
    fi
  fi
fi

echo ""
echo "Quick start:"
echo "  cat > config.json <<EOF"
echo "  {"
echo "    \"model\": \"/path/to/model.json\","
echo "    \"port\": 8080"
echo "  }"
echo "  EOF"
echo "  $BINARY_PATH config.json"
