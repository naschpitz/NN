# NN — Monorepo

NN is a C++17 + Qt + OpenCL neural-network monorepo. The components that used to
live in separate repositories (ANN, CNN, NN-CLI, NN-Server) are now **plain
directories in one git repo**. Only `OpenCLWrapper` remains a git submodule.

> This file lives at the **repo root** and is the single source of truth for the
> whole monorepo. OpenCode loads it for any work under `NN/`. Per-component notes
> are at the bottom; add a component-level `AGENTS.md` only for genuinely
> component-specific guidance. (Do **not** add it to `.gitignore` — it should
> travel with the repo.)

## Layout

```
NN/
  CMakeLists.txt     Top-level build; BUILD_ANN / BUILD_CNN / BUILD_NN_CLI / BUILD_NN_SERVER toggles
  Common/            Header-only shared config structs (Common_*.hpp); exposed as the NN_Common INTERFACE lib
  ANN/               Feedforward ANN library (CPU + GPU backends). Leaf dependency.
  CNN/               Convolutional library. Depends on ANN (PUBLIC). Reuses ANN per-sample for dense layers.
  NN-CLI/            CLI frontend: train / predict / test / calibrate. Links CNN (→ ANN, OpenCLWrapper).
  NN-Server/         HTTP inference server (QTcpServer) with a thread-safe CorePool over ANN/CNN cores.
  OpenCLWrapper/     OpenCL abstraction (OCLW_Core, OCLW_CU). Used by ANN & CNN for GPU compute. (git submodule)
```

Dependency order (PUBLIC-linked, so order matters): **OpenCLWrapper → ANN → CNN
→ {NN-CLI, NN-Server}**, with `Common/` headers available everywhere via
`#include "Common/Xxx.hpp"`.

## Build

### Primary — root `build.sh`

From the repo root, run the orchestrator script which iterates over all
components, copies a shared `CMakeUserPresets.json` into each, configures with a
cmake preset, and builds into `<component>/build/`. It also runs
`git submodule update --init --recursive` automatically.

```bash
./build.sh            # default "static" preset
./build.sh shared     # shared-library preset
```

Requires a `CMakeUserPresets.json` at the repo root with Qt6 kit paths
(git-ignored, machine-specific).

### Single component

Each component has its own `build.sh` for standalone builds:
```bash
./ANN/build.sh
./CNN/build.sh
./NN-CLI/build.sh
./NN-Server/build.sh
```

### Manual cmake (advanced)

The raw cmake commands still work for toggling components or passing custom flags
(dependencies are validated — CNN requires ANN):
```bash
cmake -B build && cmake --build build -j$(nproc)
cmake -B build -DBUILD_NN_SERVER=OFF -DBUILD_NN_CLI=ON && cmake --build build -j$(nproc)
```

Build directories (`<component>/build/`, `build*/`) are gitignored.

## Tests

There is **no CTest wiring**; each component builds a standalone test executable:

| Component | Test binary | Notes |
|-----------|-------------|-------|
| ANN       | `test_ann`       | |
| CNN       | `test_cnn`       | |
| NN-CLI    | `test_nncli`     | spawns the built `NN-CLI` binary; pass `--full` to include the long MNIST train/test cases |
| NN-Server | `test_endpoints`, `test_logger` | |

Run a test binary directly from the build dir (e.g. `./build/.../test_nncli`).
Add tests for every new feature (see principles below).

## Code organization standard

This matches the existing code — keep it consistent.

### `.hpp` — class layout
- Access order: `public:` → `protected:` → `private:` (one section each).
- Within a section: types/aliases → ctors/dtors → methods → members, each group
  under a `//-- Section Name --//` header.

### `.cpp`
- Implementations follow the **exact same order** as the `.hpp` declarations.
- Separate every method with a full-width rule:
  `//===================================================================================================================//`
- For complex files, section headers sit between two rules.
- Static member init at the top; template instantiations at the bottom.

### Naming (as used, not generic)
- Files: `<Prefix>_<Name>.{hpp,cpp}` (e.g. `CNN_CoreCPU.cpp`, `NN-CLI_Runner.cpp`).
- Namespaces: `ANN`, `CNN`, `NN_CLI` (not lowercase). Types `PascalCase`,
  methods/vars `camelCase`. Member access via `this->`.
- Formatting is enforced by `.clang-format` + pre-commit hooks — run them.

## Architecture & design principles

1. **No special-case branching on type in callers.** Adding a variant (activation,
   cost function, layer type) must not spawn a parallel function/kernel or an
   `if (type == X)` in the caller. Add a dedicated impl function and a new `case`
   inside the one existing function/switch; the decision lives *inside* the call.
2. **Readability — extract temporaries.** Never inline complex expressions
   (`.data()`, `const_cast`, indexing, member access) into call arguments; assign
   named locals first.
3. **Tests for every new feature** — unit tests in isolation **and** an
   integration test through the full pipeline (train/predict/test).
4. **Concurrency:** the CPU cores parallelize per-batch on a thread pool and
   invoke callbacks from inside that parallel region. Be careful running
   pool-using work (e.g. a validation pass) from a training callback — see the
   global `deadlock-triage` skill. Each core should use its own pool, not a
   shared global one.

## Commit conventions

Conventional commits (`feat` / `fix` / `refactor` / `perf` / `test` / `docs` /
`chore` / `build` / `ci`), imperative subject, body explaining **why**.
**Never** add a `Co-Authored-By` or any co-author trailer. Commit/push only when
asked. (Full rules: global `commit-conventions` skill.)

## Working with the global agent stack

The global OpenCode setup (`~/.config/opencode/`) provides the orchestrator,
tiered workers, the `debugger` agent, and skills (`submodule-sync`,
`deadlock-triage`, `perf-profiling`, `memory-leak-hunt`, `security-review`,
`pr-prep`, `flaky-test-stabilizer`, `commit-conventions`, `code-review`,
`dry-check`). Those are repo-agnostic; this file is the repo-specific context
that complements them. They compose — this file does not replace them.

---

## Component notes

- **ANN** — standalone feedforward library; `ANN_CoreCPU*` (CPU) and
  `ANN_CoreGPU*` (GPU) backends, activation/cost functions, device abstraction.
  Leaf dependency; the dense layers inside CNN reuse its per-sample API.
- **CNN** — convolution/pooling/normalization/residual layers; `CNN_CoreCPU*` /
  `CNN_CoreGPU*`. Depends on ANN and transitively exports it.
- **NN-CLI** — JSON-config-driven CLI; modes `train` / `predict` / `test` /
  `calibrate`; `--device cpu|gpu`; GPU image augmentation. Source in
  `NN-CLI_*Runner.*`, `main.cpp`.
- **NN-Server** — QTcpServer-based HTTP inference server; `CorePool` manages ANN/CNN core
  instances with thread-safe handling; request routing + JSON I/O; model loader
  and image pre/post-processing.
- **OpenCLWrapper** — OpenCL device/context/kernel abstraction shared by the GPU
  backends. Edit it in the submodule, then bump the pointer (see `submodule-sync`).
- **Common** — header-only shared config/result structs included as
  `Common/Common_*.hpp`; no build step (INTERFACE library `NN_Common`).
