# ANN — Feedforward Neural Network Library

C++ feedforward (MLP) library with CPU and GPU (OpenCL) backends. Pure library — no file I/O; callers (NN-CLI, NN-Server) handle serialization.

## Build

```bash
./build.sh            # from repo root (all components)
./ANN/build.sh        # standalone
```

Builds into `ANN/build/`. Test binary: `test_ann`.

## Architecture

- `Core` (abstract) → `CoreCPU` / `CoreGPU` (concrete backends)
- Each backend uses its own `QThreadPool` per-core — never a shared global pool
- `CoreCPUWorker` / `CoreGPUWorker` handle the actual compute
- GPU paths use `ANN_GPUBufferManager` and `ANN_GPUKernelBuilder` for OpenCL buffer/kernel lifecycle
- Dense layers (fully-connected) inside CNN reuse ANN's per-sample API

## Testing

```bash
cd ANN/build && ./test_ann          # all tests
```

Tests cover: CPU basic, GPU basic, features, exact numerical matching, serialization.

## Gotchas

- `numThreads = 0` means "use all cores"; `numGPUs = 0` means "use all GPUs"
- ANN handles dense (fully-connected) layers; CNN wraps ANN for its dense head
- No built-in serialization — model save/load is NN-CLI/NN-Server's responsibility
