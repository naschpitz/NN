# NN

A C++17 + Qt6 + OpenCL neural-network monorepo: feedforward (ANN) and
convolutional (CNN) libraries with CPU and GPU backends, a command-line tool
(NN-CLI), and an HTTP inference server (NN-Server).

## Layout

```
ANN/            Feedforward ANN library (CPU + GPU). Leaf dependency.
CNN/            Convolutional library; depends on ANN.
Common/         Header-only shared config/result structs (NN_Common INTERFACE lib).
NN-CLI/         CLI: train / predict / test / calibrate.
NN-Server/      HTTP inference server (QTcpServer) over a thread-safe CorePool.
ANN/extern/OpenCLWrapper   OpenCL abstraction (git submodule).
CNN/extern/OpenCLWrapper   OpenCL abstraction (git submodule).
```

ANN and CNN consume OpenCLWrapper directly (each carries it under `extern/`);
NN-CLI and NN-Server get it transitively. Dependency order:
**OpenCLWrapper → ANN → CNN → {NN-CLI, NN-Server}**.

## Prerequisites

- A C++17 compiler (GCC ≥ 9 or Clang ≥ 10)
- CMake ≥ 3.14
- **Qt6** (Core, Concurrent; NN-Server also Network)
- **OpenCL** ICD + headers, plus a working GPU driver/runtime for the GPU paths
- **ncurses (wide)** for the NN-CLI training UI
- *Optional:* tcmalloc — NN-CLI links it if present (lower RSS under heavy augmentation)

Install on common systems:

| OS | Command |
|----|---------|
| Debian/Ubuntu | `sudo apt install build-essential cmake qt6-base-dev ocl-icd-opencl-dev libncurses-dev libgoogle-perftools-dev` |
| Fedora | `sudo dnf install gcc-c++ cmake qt6-qtbase-devel ocl-icd-devel ncurses-devel gperftools-devel` |
| Arch | `sudo pacman -S base-devel cmake qt6-base ocl-icd ncurses gperftools` |
| macOS | `brew install cmake qt6 ncurses gperftools` (OpenCL ships with macOS) |

A distro/Homebrew Qt6 installs onto CMake's default search paths, so the build
needs **no extra configuration**. If your Qt6 lives somewhere non-standard, point
CMake at it — see [Build](#build) below.

## Build

Each component builds into **its own `build/` directory** (there is no single
root build dir by default). A component is built standalone — it pulls its
dependencies via `add_subdirectory` (`CNN`→`ANN`→`extern/OpenCLWrapper`, …) — so
building `NN-CLI` or `NN-Server` transitively builds the libraries they need.

```bash
git clone <repo-url> NN && cd NN
git submodule update --init --recursive   # OpenCLWrapper under each component's extern/
```

### Build everything — `./build.sh`

```bash
./build.sh            # static Qt6 (default)
./build.sh shared     # shared Qt6
```

`build.sh` builds ANN, CNN, NN-CLI and NN-Server, each into its own `build/`. It
reads the Qt6 kit paths from the repo-root **`CMakeUserPresets.json`** (git-ignored,
machine-specific) and copies that master into each component before building, so the
paths are maintained in one place. Create it once with your kit locations:

```json
{
  "version": 3,
  "cmakeMinimumRequired": { "major": 3, "minor": 21 },
  "configurePresets": [
    { "name": "static", "binaryDir": "${sourceDir}/build",
      "cacheVariables": { "CMAKE_PREFIX_PATH": "/path/to/Qt6-static", "CMAKE_BUILD_TYPE": "Release" } },
    { "name": "shared", "binaryDir": "${sourceDir}/build",
      "cacheVariables": { "CMAKE_PREFIX_PATH": "/path/to/Qt6-shared", "CMAKE_BUILD_TYPE": "Release" } }
  ],
  "buildPresets": [
    { "name": "static", "configurePreset": "static" },
    { "name": "shared", "configurePreset": "shared" }
  ]
}
```

### Build a single component

If Qt6 is on the default paths (distro/Homebrew `qt6-base-dev`) — no preset needed:

```bash
cd NN-CLI && cmake -B build && cmake --build build -j"$(nproc)"   # → NN-CLI/build/
```

For a custom/non-standard Qt6, either pass `-DCMAKE_PREFIX_PATH=/path/to/Qt6`,
`export CMAKE_PREFIX_PATH=/path/to/Qt6` once in your shell, or — once the master
preset has been distributed (e.g. by `build.sh`) — `cmake --preset static`.

### Static vs shared, and the optional unified build

A *static* Qt6 kit yields a self-contained binary (no runtime Qt dependency) —
good for shipping, but a static kit isn't a stock package. *Shared* system Qt6 is
the lower-friction default for development. An all-in-one unified build is still
available at the repo root (`cmake -B build` builds every component via
`-DBUILD_ANN`/`-DBUILD_CNN`/`-DBUILD_NN_CLI`/`-DBUILD_NN_SERVER`), but the
per-component layout above is the default.

## Tests

Each component builds a standalone test executable (no CTest wiring); it lives in
that component's `build/`:

| Component | Binary | Notes |
|-----------|--------|-------|
| ANN | `ANN/build/test_ann` | |
| CNN | `CNN/build/test_cnn` | |
| NN-CLI | `NN-CLI/build/test_nncli` | pass `--full` to include the long MNIST train/test cases |
| NN-Server | `NN-Server/build/test_endpoints`, `test_logger` | |

Run a test binary directly, e.g. `./NN-CLI/build/test_nncli`.
