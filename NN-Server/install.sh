#!/bin/bash
#
# NN-Server installer.
#
# Clones the NN monorepo from GitHub, builds the NN-Server release binary, runs
# an interactive configuration interview, writes config.json, and optionally
# installs a systemd service so NN-Server starts on boot.
#
# Download & run in one step (recommended):
#   curl -fsSL https://raw.githubusercontent.com/naschpitz/NN/main/NN-Server/install.sh | bash
#
# Or save it first:
#   curl -fsSL <url> -o install.sh && chmod +x install.sh && ./install.sh [install-path]
#
# Re-running the script detects an existing installation (tracked in a hidden
# state file under ~/.config/nn-server/) and offers to update it to the latest
# tagged release.

set -euo pipefail

#-----------------------------------------------------------------------------#
# Constants
#-----------------------------------------------------------------------------#
REPO_URL="https://github.com/naschpitz/NN.git"
RAW_URL="https://raw.githubusercontent.com/naschpitz/NN/main/NN-Server/install.sh"
STATE_FILE="${HOME}/.config/nn-server/install.state"
SERVICE_NAME="nn-server"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
BUILD_FLAGS="--release"               # shipped binary is a Release build
DEFAULT_INSTALL_PATH="${HOME}/nn-server"

# Globals populated during the run.
INSTALL_PATH=""
REPO_PATH=""
BINARY_PATH=""
SYMLINK_PATH=""
CONFIG_PATH=""

#-----------------------------------------------------------------------------#
# Helpers
#-----------------------------------------------------------------------------#

banner() {
    echo
    echo "=================================================================="
    echo "  $1"
    echo "=================================================================="
}

# Log to stderr so stdout stays clean for functions whose return value is
# captured via $(...).
log() { echo "$@" >&2; }

die() { echo "ERROR: $1" >&2; exit 1; }

usage() {
    cat <<EOF
Usage: $0 [install-path]

  Clones the NN monorepo, builds NN-Server (release), runs a config interview,
  and optionally installs a systemd service.

  [install-path]  Optional. Directory to install into
                  (default: ${DEFAULT_INSTALL_PATH}).
                  Re-running detects an existing install and offers to update it.

Download & run in one step:
  curl -fsSL ${RAW_URL} | bash
EOF
}

# Read a value from the user into the named variable.
# Works under 'curl | bash' (where stdin is the script body) by reading from
# /dev/tty. Pressing Enter accepts the [default] when one is given.
#   prompt <var> <text> [default]
prompt() {
    local var="$1" text="$2" default="${3:-}" full val=""
    if [ -n "$default" ]; then
        full="${text} [${default}] "
    else
        full="${text} "
    fi
    if [ -t 0 ]; then
        read -rp "$full" val || true
    else
        read -rp "$full" val </dev/tty || true
    fi
    if [ -z "$val" ] && [ -n "$default" ]; then val="$default"; fi
    printf -v "$var" '%s' "$val"
}

# Yes/no question. Returns 0 for yes, 1 for no.
#   ask_yesno <text> [y|n default]
ask_yesno() {
    local text="$1" def="${2:-n}" hint ans=""
    if [ "$def" = "y" ]; then hint="[Y/n]"; else hint="[y/N]"; fi
    if [ -t 0 ]; then
        read -rp "${text} ${hint} " ans || true
    else
        read -rp "${text} ${hint} " ans </dev/tty || true
    fi
    [ -z "$ans" ] && ans="$def"
    [[ "$ans" =~ ^[Yy] ]]
}

# Escape a string for embedding inside JSON double quotes.
json_str() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    s="${s//$'\n'/\\n}"
    s="${s//$'\r'/\\r}"
    s="${s//$'\t'/\\t}"
    printf '%s' "$s"
}

# Escape a string for a single-quoted shell value (used in the state file).
state_quote() {
    local s="$1"
    s="${s//\'/\'\\\'\'}"
    printf '%s' "$s"
}

#-----------------------------------------------------------------------------#
# Dependencies
#-----------------------------------------------------------------------------#

install_dependencies() {
    local cmd=""
    if   command -v apt    >/dev/null 2>&1; then cmd="sudo apt install -y git cmake make g++ qt6-base-dev ocl-icd-opencl-dev"
    elif command -v dnf    >/dev/null 2>&1; then cmd="sudo dnf install -y git cmake make gcc-c++ qt6-qtbase-devel ocl-icd-devel"
    elif command -v pacman >/dev/null 2>&1; then cmd="sudo pacman -S --noconfirm git cmake make gcc qt6-base ocl-icd"
    elif command -v zypper >/dev/null 2>&1; then cmd="sudo zypper install -y git cmake make gcc-c++ qt6-base-devel ocl-icd-devel"
    fi

    if [ -z "$cmd" ]; then
        echo "  Could not detect a package manager."
        echo "  Please install manually: git, cmake, make, g++ (C++20),"
        echo "  Qt6 (Core/Concurrent/Network), and OpenCL ICD + headers."
        die "Install the missing dependencies and re-run this script."
    fi

    echo "  Detected package manager. Proposed command:"
    echo "    ${cmd}"
    if ! ask_yesno "Install the missing dependencies now?" "y"; then
        die "Required dependencies were not installed."
    fi
    bash -c "$cmd"

    # Re-verify.
    local still=() c
    for c in git cmake make g++; do command -v "$c" >/dev/null 2>&1 || still+=("$c"); done
    if ! command -v qmake6 >/dev/null 2>&1 && ! command -v qmake >/dev/null 2>&1; then
        still+=("qmake6")
    fi
    [ ${#still[@]} -eq 0 ] || die "Some dependencies are still missing: ${still[*]}"
}

check_dependencies() {
    banner "Checking dependencies"
    local missing=() c
    for c in git cmake make g++; do command -v "$c" >/dev/null 2>&1 || missing+=("$c"); done
    if ! command -v qmake6 >/dev/null 2>&1 && ! command -v qmake >/dev/null 2>&1; then
        missing+=("qt6 (qmake6)")
    fi
    if [ ${#missing[@]} -eq 0 ]; then
        echo "  All required build tools found."
        return 0
    fi
    echo "  Missing: ${missing[*]}"
    install_dependencies
    echo "  Dependencies installed."
}

#-----------------------------------------------------------------------------#
# Repository / build
#-----------------------------------------------------------------------------#

# Print the highest semver tag in the repo (or empty if there are none).
latest_tag() {
    git -C "$REPO_PATH" tag --sort=-v:refname 2>/dev/null | sed -n '1p' | tr -d '[:space:]'
}

# True if $1 is a strictly greater version than $2 (v-prefix tolerant, semver-ish).
version_gt() {
    [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -n1)" = "$1" ]
}

clone_or_fetch() {
    if [ -d "$REPO_PATH/.git" ]; then
        echo "  Existing repository clone found at $REPO_PATH"
        git -C "$REPO_PATH" fetch --tags --quiet
    else
        echo "  Cloning ${REPO_URL} -> $REPO_PATH"
        git clone --quiet "$REPO_URL" "$REPO_PATH"
    fi
    git -C "$REPO_PATH" submodule update --init --recursive --quiet
}

# Check out the latest release tag (prints the tag, or the short SHA if no tags).
checkout_latest_release() {
    local tag
    tag="$(latest_tag)"
    if [ -n "$tag" ]; then
        log "  Checking out latest release tag: $tag"
        git -C "$REPO_PATH" checkout --quiet "$tag"
        git -C "$REPO_PATH" submodule update --init --recursive --quiet
    else
        log "  No release tags found; staying on default branch."
        tag="$(git -C "$REPO_PATH" rev-parse --short HEAD)"
    fi
    printf '%s' "$tag"
}

build_server() {
    banner "Building NN-Server (release)"
    ( cd "$REPO_PATH" && bash NN-Server/build.sh $BUILD_FLAGS )
    BINARY_PATH="$REPO_PATH/NN-Server/build/NN-Server"
    [ -x "$BINARY_PATH" ] || die "Build finished but binary not found at $BINARY_PATH"
    echo "  Built: $BINARY_PATH"
}

make_symlink() {
    SYMLINK_PATH="$INSTALL_PATH/${SERVICE_NAME}"
    ln -sfn "$BINARY_PATH" "$SYMLINK_PATH"
    echo "  Symlink: $SYMLINK_PATH -> $BINARY_PATH"
}

#-----------------------------------------------------------------------------#
# Config interview (only on fresh install, or when config.json is missing)
#-----------------------------------------------------------------------------#

interview_config() {
    banner "Configuration interview"
    cat <<MSG
  NN-Server reads its settings from config.json. Answer the following questions;
  press Enter to accept the [default] shown. You can edit config.json later.
MSG

    local model port pool_size max_body max_queue log_file max_log_size nproc
    nproc="$(nproc 2>/dev/null || echo 4)"

    prompt model \
"  Path to your trained model (.nnmodel or .nnmodel.tar) that NN-Server will serve.
  Leave empty if you don't have one yet (server starts, but /predict errors until set)" \
        ""

    prompt port \
"  TCP port the HTTP server listens on (1-65535)" "8080"
    while ! [[ "$port" =~ ^[0-9]+$ ]] || [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; do
        prompt port "  Invalid. Enter a port between 1 and 65535" "8080"
    done

    prompt pool_size \
"  Number of pre-loaded Core instances in the thread pool
  (more = more concurrent predictions but more RAM)" "$nproc"
    while ! [[ "$pool_size" =~ ^[0-9]+$ ]] || [ "$pool_size" -lt 1 ]; do
        prompt pool_size "  Invalid. Enter a positive integer" "$nproc"
    done

    prompt max_body \
"  Max accepted request body size in MB (0 = unlimited)" "10"
    while ! [[ "$max_body" =~ ^[0-9]+$ ]]; do
        prompt max_body "  Invalid. Enter a non-negative integer" "10"
    done

    prompt max_queue \
"  Max concurrent in-flight requests; extras get HTTP 503 (0 = unlimited)" "0"
    while ! [[ "$max_queue" =~ ^[0-9]+$ ]]; do
        prompt max_queue "  Invalid. Enter a non-negative integer" "0"
    done

    local default_log="$INSTALL_PATH/nn-server.log"
    prompt log_file \
"  Path to the log file (leave empty to disable file logging;
  systemd also captures output to journald)" "$default_log"

    prompt max_log_size \
"  Max log file size in GB before it rotates (0 = unlimited)" "1"
    while ! [[ "$max_log_size" =~ ^[0-9]+$ ]]; do
        prompt max_log_size "  Invalid. Enter a non-negative integer" "1"
    done

    CONFIG_PATH="$INSTALL_PATH/config.json"
    cat > "$CONFIG_PATH" <<EOF
{
  "model": "$(json_str "$model")",
  "port": $port,
  "poolSize": $pool_size,
  "maxBodySize": $max_body,
  "maxQueueSize": $max_queue,
  "logFile": "$(json_str "$log_file")",
  "maxLogSize": $max_log_size
}
EOF
    echo "  Wrote $CONFIG_PATH"
}

#-----------------------------------------------------------------------------#
# State file (hidden) — tracks version + install location for updates
#-----------------------------------------------------------------------------#

save_state() {
    local version="$1"
    mkdir -p "$(dirname "$STATE_FILE")"
    cat > "$STATE_FILE" <<EOF
# NN-Server install state (managed by install.sh). Safe to remove to force a
# fresh install, but do not commit or share this file.
VERSION='$(state_quote "$version")'
INSTALL_PATH='$(state_quote "$INSTALL_PATH")'
REPO_PATH='$(state_quote "$REPO_PATH")'
BINARY_PATH='$(state_quote "$BINARY_PATH")'
SYMLINK_PATH='$(state_quote "$SYMLINK_PATH")'
CONFIG_PATH='$(state_quote "$CONFIG_PATH")'
BUILD_FLAGS='$(state_quote "$BUILD_FLAGS")'
INSTALLED_AT='$(date -u +%Y-%m-%dT%H:%M:%SZ)'
EOF
    chmod 600 "$STATE_FILE"
    echo "  State saved to $STATE_FILE"
}

# Load state if it exists AND the recorded repo is still present.
load_state() {
    [ -f "$STATE_FILE" ] || return 1
    # shellcheck disable=SC1090
    source "$STATE_FILE"
    [ -n "${INSTALL_PATH:-}" ] && [ -d "${REPO_PATH:-}/.git" ]
}

#-----------------------------------------------------------------------------#
# systemd
#-----------------------------------------------------------------------------#

have_systemd() {
    [ -d /run/systemd/system ] || grep -qa systemd /proc/1/comm 2>/dev/null
}

write_service_unit() {
    local user group
    user="$(id -un)"
    group="$(id -gn)"
    sudo tee "$SERVICE_FILE" >/dev/null <<EOF
[Unit]
Description=NN-Server Neural Network Inference Server
After=network.target

[Service]
Type=simple
User=$user
Group=$group
ExecStart=$SYMLINK_PATH --config $CONFIG_PATH
WorkingDirectory=$INSTALL_PATH
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
}

# Offer/refresh the systemd service. Asked on fresh installs and whenever the
# service is missing — even during an update.
setup_systemd() {
    banner "systemd service"

    if ! have_systemd; then
        echo "  This system does not appear to use systemd; skipping auto-start setup."
        echo "  Start NN-Server manually:  $SYMLINK_PATH --config $CONFIG_PATH"
        return 0
    fi

    if [ -f "$SERVICE_FILE" ]; then
        echo "  Existing service found at $SERVICE_FILE."
        if ask_yesno "Refresh the unit file and restart the service now?" "y"; then
            write_service_unit
            sudo systemctl daemon-reload
            if sudo systemctl restart "$SERVICE_NAME"; then
                echo "  Service refreshed and restarted."
            else
                echo "  Service did not start cleanly. Check: journalctl -u $SERVICE_NAME -e"
            fi
        fi
        return 0
    fi

    echo "  No '$SERVICE_NAME' service is installed."
    if ! ask_yesno "Create a systemd service so NN-Server starts automatically on boot?" "y"; then
        echo "  Skipped. Start manually:  $SYMLINK_PATH --config $CONFIG_PATH"
        return 0
    fi

    write_service_unit
    sudo systemctl daemon-reload
    sudo systemctl enable "$SERVICE_NAME" >/dev/null
    if [ -f "$CONFIG_PATH" ]; then
        if sudo systemctl start "$SERVICE_NAME"; then
            echo "  Service created, enabled, and started."
        else
            echo "  Service created and enabled, but failed to start."
            echo "  Check: journalctl -u $SERVICE_NAME -e"
        fi
    else
        echo "  Service created and enabled (starts on boot once $CONFIG_PATH exists)."
    fi
}

#-----------------------------------------------------------------------------#
# Flows
#-----------------------------------------------------------------------------#

fresh_flow() {
    banner "Fresh installation"
    local path_arg="${1:-}"
    if [ -n "$path_arg" ]; then
        INSTALL_PATH="$path_arg"
    else
        prompt INSTALL_PATH \
"  Where should NN-Server be installed?
  (the repo will be cloned into a subdirectory of this path)" \
            "$DEFAULT_INSTALL_PATH"
    fi

    mkdir -p "$INSTALL_PATH"
    INSTALL_PATH="$(cd "$INSTALL_PATH" && pwd)"   # resolve to absolute
    REPO_PATH="$INSTALL_PATH/NN"

    clone_or_fetch
    local version
    version="$(checkout_latest_release)"
    build_server
    make_symlink

    if [ -f "$INSTALL_PATH/config.json" ]; then
        CONFIG_PATH="$INSTALL_PATH/config.json"
        echo "  Existing config.json kept: $CONFIG_PATH"
    else
        interview_config
    fi

    save_state "$version"
}

update_flow() {
    banner "Existing installation detected"
    echo "  Version:     ${VERSION:-unknown}"
    echo "  Install dir: $INSTALL_PATH"
    echo "  Repo:        $REPO_PATH"

    REPO_PATH="${REPO_PATH:-$INSTALL_PATH/NN}"
    BINARY_PATH="${BINARY_PATH:-$REPO_PATH/NN-Server/build/NN-Server}"
    SYMLINK_PATH="${SYMLINK_PATH:-$INSTALL_PATH/$SERVICE_NAME}"
    CONFIG_PATH="${CONFIG_PATH:-$INSTALL_PATH/config.json}"

    # Repo gone -> offer re-clone into the recorded location.
    if [ ! -d "$REPO_PATH/.git" ]; then
        echo "  The recorded repository is missing ($REPO_PATH)."
        if ! ask_yesno "Re-clone into the same location and reinstall?" "y"; then
            die "Cannot update: repository missing."
        fi
        clone_or_fetch
        local version
        version="$(checkout_latest_release)"
        build_server
        make_symlink
        [ -f "$CONFIG_PATH" ] || interview_config
        save_state "$version"
        return
    fi

    echo "  Checking for updates..."
    git -C "$REPO_PATH" fetch --tags --quiet
    local latest version_current
    latest="$(latest_tag)"
    version_current="${VERSION:-}"

    if [ -n "$latest" ] && version_gt "$latest" "$version_current"; then
        echo "  Update available: ${version_current:-unknown} -> $latest"
        if ask_yesno "Update to $latest now?" "y"; then
            git -C "$REPO_PATH" checkout --quiet "$latest"
            git -C "$REPO_PATH" submodule update --init --recursive --quiet
            build_server
            make_symlink
            save_state "$latest"
            echo "  Updated to $latest."
        else
            echo "  Skipping update."
        fi
    else
        echo "  Already up to date (${version_current:-${latest:-unknown}})."
        if ask_yesno "Rebuild from the current source anyway?" "n"; then
            build_server
            make_symlink
            save_state "${latest:-$version_current}"
        fi
    fi
}

final_summary() {
    banner "Done"
    echo "  Binary:  $BINARY_PATH"
    echo "  Run:     $SYMLINK_PATH --config $CONFIG_PATH"
    echo "  Config:  $CONFIG_PATH"
    if have_systemd && command -v systemctl >/dev/null 2>&1; then
        echo "  Service: sudo systemctl status $SERVICE_NAME"
    fi
    echo
    echo "  IMPORTANT: do not delete $REPO_PATH"
    echo "             (it holds the OpenCL kernels needed for GPU inference)."
}

#-----------------------------------------------------------------------------#
# Main
#-----------------------------------------------------------------------------#

main() {
    if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
        usage
        exit 0
    fi

    banner "NN-Server installer"

    check_dependencies

    if load_state; then
        update_flow
    else
        fresh_flow "${1:-}"
    fi

    setup_systemd
    final_summary
}

main "$@"
