# NN — Monorepo

C++17 + Qt6 + OpenCL neural-network monorepo: feedforward (ANN) and convolutional (CNN)
libraries with CPU/GPU backends, a CLI tool, and an HTTP inference server.

## Layout

```
NN/
  CMakeLists.txt     Top-level build toggle (BUILD_ANN / BUILD_CNN / BUILD_NN_CLI / BUILD_NN_SERVER)
  Common/            Header-only config structs (Common_*.hpp); NN_Common INTERFACE lib
  ANN/               Feedforward library (CPU + GPU). Leaf dependency.
  CNN/               Convolutional library; depends on ANN.
  NN-CLI/            CLI: train / predict / test / calibrate.
  NN-Server/         HTTP inference server (QTcpServer + CorePool).
  ANN/extern/OpenCLWrapper   OpenCL abstraction (git submodule).
  CNN/extern/OpenCLWrapper   OpenCL abstraction (git submodule).
```

Dependency order (PUBLIC-linked): **OpenCLWrapper → ANN → CNN → {NN-CLI, NN-Server}**.
Common headers: `#include "Common/Xxx.hpp"` from anywhere.

## Build

### All components
```bash
./build.sh            # static Qt6 (default)
./build.sh shared     # shared Qt6
```
Copies root `CMakeUserPresets.json` into each component, runs `git submodule update`, then builds each into `<component>/build/`.

### Single component
```bash
./ANN/build.sh
./CNN/build.sh
./NN-CLI/build.sh
./NN-Server/build.sh
```

### Manual cmake
```bash
cmake -B build && cmake --build build -j$(nproc)
# Toggle components: cmake -B build -DBUILD_NN_SERVER=OFF -DBUILD_NN_CLI=ON && cmake --build build -j$(nproc)
```
Dependencies validated at configure time (CNN requires ANN).

### Prerequisites
- C++17 compiler (GCC ≥ 9 / Clang ≥ 10)
- CMake ≥ 3.14
- Qt6 (Core, Concurrent; NN-Server needs Network)
- OpenCL ICD + headers + GPU driver
- ncurses-wide (NN-CLI training UI)
- *Optional:* tcmalloc (lower RSS under heavy augmentation)

`CMakeUserPresets.json` at the repo root holds Qt6 kit paths (gitignored, machine-specific).

## Tests

No CTest wiring. Each component builds its own test binary:

| Component | Binary | Notes |
|-----------|--------|-------|
| ANN | `test_ann` | |
| CNN | `test_cnn` | |
| NN-CLI | `test_nncli` | pass `--full` for long MNIST train/test |
| NN-Server | `test_endpoints`, `test_logger` | |

Run from the component's `build/` dir. Add tests for every new feature.

## Code style

Enforced by `.clang-format` (LLVM-based, 2-space indent, 120-col limit, Linux brace style) + pre-commit hooks.
**Match existing style in every file — even if you'd do it differently.**

### `.hpp` class layout
`public:` → `protected:` → `private:`. Within each: types → ctors/dtors → methods → members.
Each sub-group under `//-- Section Name --//`.

### `.cpp`
Implementations follow the `.hpp` order exactly. Separate every method with:
`//===================================================================================================================//`
Static member init at top; template instantiations at bottom.

### Naming
- Files: `<Prefix>_<Name>.{hpp,cpp}` (e.g. `CNN_CoreCPU.cpp`, `NN-CLI_Runner.cpp`)
- Namespaces: `ANN`, `CNN`, `NN_CLI` (PascalCase, not lowercase)
- Types: PascalCase · methods/vars: camelCase · member access: `this->`

## Principles

### Design

1. **No type branching in callers.** Adding a variant (activation, cost, layer) means one new `case` in the existing dispatch function — no parallel paths, no `if (type == X)` in callers.
2. **Extract temporaries.** Never inline `.data()`, `const_cast`, indexing into call arguments; use named locals.
3. **Tests for every feature.** Unit tests + integration through the full pipeline (train → predict → test).
4. **Concurrency.** CPU cores parallelize per-batch on a thread pool, invoking callbacks inside that region. Each core uses its own pool — never a shared global one. Running pool-using work from a training callback risks deadlock (see `deadlock-triage` skill).

### Behavior

#### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

#### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

#### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

#### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

## Commit conventions

Conventional commits (`feat` / `fix` / `refactor` / `perf` / `test` / `docs` / `chore` / `build` / `ci`).
Imperative subject, body explains **why**. **Never** add `Co-Authored-By`. Commit/push only when asked.

## Pre-commit hooks

- **clang-format** (v18.1.8) — excludes `extern/` and `libs/`
- **Local blank-line script** (`scripts/ensure-blank-lines.py`) — ensures blank lines around if/else/try/catch in C/C++ — excludes `extern/` and `libs/`

## OpenCLWrapper

Submodule at `ANN/extern/OpenCLWrapper` and `CNN/extern/OpenCLWrapper` (same remote, two working trees).
Edit in the submodule, then bump the pointer (see `submodule-sync` skill).
Clang-format and blank-line hooks exclude `extern/` — submodule files are not reformatted.

## Component notes

- **ANN** — feedforward library; `ANN_CoreCPU*` / `ANN_CoreGPU*` backends, activation/cost functions, device abstraction. Leaf dependency. → `ANN/AGENTS.md`
- **CNN** — conv/pool/norm/residual layers; `CNN_CoreCPU*` / `CNN_CoreGPU*`. Depends on ANN, transitively exports it. → `CNN/AGENTS.md`
- **NN-CLI** — JSON-config CLI; modes `train`/`predict`/`test`/`calibrate`; `--device cpu|gpu`. Sources in `NN-CLI_*Runner.*`, `main.cpp`. → `NN-CLI/AGENTS.md`
- **NN-Server** — QTcpServer HTTP server; `CorePool` manages ANN/CNN instances thread-safely; request routing, JSON I/O, model loader, image processing. → `NN-Server/AGENTS.md`
- **Common** — header-only config/result structs (`Common/Common_*.hpp`); INTERFACE library `NN_Common`.
