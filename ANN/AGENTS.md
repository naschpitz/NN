# ANN — Feedforward Neural Network Library

C++ feedforward (MLP) library with CPU and GPU (OpenCL) backends. Pure library — no file I/O; callers (NN-CLI, NN-Server) handle serialization.

## Build

```bash
./build.sh            # from repo root (all components)
./ANN/build.sh        # standalone
```

Builds into `ANN/build/`. Test binary: `test_ann`.

## Dependencies

- Qt6 Core + Concurrent
- OpenCLWrapper (submodule at `extern/OpenCLWrapper`)
- Common/ headers via `#include "Common/Xxx.hpp"` (resolved from monorepo parent dir)

## Public API

All interaction via `ANN::Core<float>` factory:

```cpp
#include <ANN_Core.hpp>
#include <ANN_CoreConfig.hpp>
#include <ANN_Sample.hpp>
#include <ANN_SampleProvider.hpp>

auto core = ANN::Core<float>::makeCore(config);
core->train(numSamples, ANN::makeSampleProvider(samples));
auto result = core->test(numSamples, ANN::makeSampleProvider(testSamples));
// core->predict(...) for inference
```

CoreConfig fields: `modeType`, `deviceType`, `numThreads`, `numGPUs`, `layersConfig`, `costFunctionConfig`, `trainingConfig`, `testConfig`, `parameters` (pre-trained weights), `progressReports`, `logLevel`, `loadedEpochHistory`.

## Key types

- `ANN::Input<T>` → `std::vector<T>` (1D vector)
- `ANN::Inputs<T>` → `std::vector<Input<T>>` (batch)
- `ANN::Output<T>` → `std::vector<T>`
- `ANN::Tensor1D/2D/3D<T>` → nested `std::vector`
- `ANN::Logits<T>` → pre-activation z values (for OOD detection)

## Architecture

- `Core` (abstract) → `CoreCPU` / `CoreGPU` (concrete backends)
- Each backend uses its own `QThreadPool` per-core — never a shared global pool
- `CoreCPUWorker` / `CoreGPUWorker` handle the actual compute
- GPU paths use `ANN_GPUBufferManager` and `ANN_GPUKernelBuilder` for OpenCL buffer/kernel lifecycle
- Dense layers (fully-connected) inside CNN reuse ANN's per-sample API

## GPU paths

- OpenCL kernels live in `opencl/*.cl`
- `ANN_GPUBufferManager` handles device/host buffer allocation
- `ANN_GPUKernelBuilder` compiles and caches OpenCL programs
- Multi-GPU: `CoreGPUWorker` partitions work across `numGPUs` devices

## Testing

```bash
cd ANN/build && ./test_ann          # all tests
```

Tests cover: CPU basic, GPU basic, features, exact numerical matching, serialization.

## Gotchas

- `numThreads = 0` means "use all cores"; `numGPUs = 0` means "use all GPUs"
- ANN handles dense (fully-connected) layers; CNN wraps ANN for its dense head
- No built-in serialization — model save/load is NN-CLI/NN-Server's responsibility
