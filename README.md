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
NN-Server/      HTTP inference server (Civetweb) over a thread-safe CorePool.
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
needs **no extra configuration**. If your Qt6 lives somewhere non-standard, see
[Custom Qt6 location](#custom-qt6-location).

## Build

```bash
git clone <repo-url> NN && cd NN
git submodule update --init --recursive     # fetches OpenCLWrapper under each extern/
cmake -B build
cmake --build build -j"$(nproc)"
```

Select components with `-DBUILD_ANN`, `-DBUILD_CNN`, `-DBUILD_NN_CLI`,
`-DBUILD_NN_SERVER` (all `ON` by default; CNN requires ANN). A component can also
be built standalone from its own directory (e.g. `cd CNN && cmake -B build && …`).

### Custom Qt6 location

If `find_package(Qt6)` can't find Qt6 (it isn't on the default paths), point CMake
at your kit:

```bash
cmake -B build -DCMAKE_PREFIX_PATH=/path/to/Qt6
```

To avoid passing it every time, either `export CMAKE_PREFIX_PATH=/path/to/Qt6` in
your shell profile, or create a **`CMakeUserPresets.json`** (git-ignored, machine-local)
pinning the path and build dir, then `cmake --preset <name>`.

### Static / self-contained binary (optional)

Building against a *static* Qt6 kit yields a binary with no runtime Qt dependency
— handy for shipping. Static Qt6 is not a stock distro package; you must supply a
static kit and point at it:

```bash
cmake -B build-static -DCMAKE_PREFIX_PATH=/path/to/static-qt6
cmake --build build-static -j"$(nproc)"
```

For everyday development, a shared system Qt6 (above) is the lower-friction choice.

## Tests

Each component builds a standalone test executable (no CTest wiring):

| Component | Binary | Notes |
|-----------|--------|-------|
| ANN | `build/ANN/test_ann` | |
| CNN | `build/CNN/test_cnn` | |
| NN-CLI | `build/NN-CLI/test_nncli` | pass `--full` to include the long MNIST train/test cases |
| NN-Server | `build/NN-Server/test_endpoints`, `test_logger` | |

Run a test binary directly, e.g. `./build/NN-CLI/test_nncli`.
