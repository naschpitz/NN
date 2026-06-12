# CNN — Convolutional Neural Network Library

C++ CNN library (conv/pool/norm/residual layers) with CPU and GPU (OpenCL) backends. Delegates dense layers to ANN.

## Build

```bash
./build.sh            # from repo root (all components)
./CNN/build.sh        # standalone
```

Builds into `CNN/build/`. Test binary: `test_cnn`.

## Dependencies

- Qt6 Core + Concurrent
- ANN (PUBLIC dependency — CNN transitively exports it)
- OpenCLWrapper (submodule at `extern/OpenCLWrapper`)
- Common/ headers via `#include "Common/Xxx.hpp"`

## Public API

```cpp
#include <CNN_Core.hpp>
#include <CNN_CoreConfig.hpp>
#include <CNN_Sample.hpp>
#include <CNN_SampleProvider.hpp>

CNN::CoreConfig<float> config;
config.inputShape = {1, 28, 28};  // C, H, W (NCHW layout)
config.layersConfig.cnnLayers = {
  {CNN::LayerType::CONV, {8, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID}},
  {CNN::LayerType::RELU, {}},
  {CNN::LayerType::POOL, {CNN::PoolTypeEnum::MAX, 2, 2, 2, 2}},
  {CNN::LayerType::FLATTEN, {}}
};
config.layersConfig.denseLayers = {{128, ANN::ActvFuncType::RELU}, {10, ANN::ActvFuncType::SIGMOID}};

auto core = CNN::Core<float>::makeCore(config);
core->train(numSamples, CNN::makeSampleProvider(samples));
```

## Key types

- `CNN::Input<T>` → `Tensor3D<T>` (NCHW 3D tensor)
- `CNN::Output<T>` → `std::vector<T>` (1D, from dense head)
- `CNN::Shape3D` — shape with C, H, W fields
- `CNN::Tensor3D<T>` → `std::vector<std::vector<std::vector<T>>>`

## Layer types

| Type | Key fields |
|------|-----------|
| CONV | numFilters, filterH/W, strideY/X, slidingStrategy |
| RELU | — |
| POOL | poolType (MAX/AVG), poolH/W, strideY/X |
| FLATTEN | — |
| GLOBALAVGPOOL | — |
| GLOBALDUALPOOL | — |
| INSTANCE_NORM | gamma, beta, epsilon |
| BATCH_NORM | — |
| RESIDUAL_START / RESIDUAL_END | — (markers; projection weights handled separately) |

## Architecture

- `Core` (abstract) → `CoreCPU` / `CoreGPU`
- `CoreCPUWorker` / `CoreGPUWorker` handle per-batch parallelism
- `CNN_GPUKernelBuilder` compiles OpenCL kernels from `opencl/*.cl`
- `CNN_CoreGPUWorkerConfig` handles GPU kernel configuration
- Dense layers (after flatten) are delegated to `ANN::Core<float>` per-sample

## GPU paths

- Kernels in `opencl/*.cl`: Propagate, Backpropagate, Update, Bridge, Normalization, GlobalAvgPool, GlobalDualPool, Residual, GEMM, Im2Col
- Multi-GPU training supported via `config.numGPUs`

## Testing

```bash
cd CNN/build && ./test_cnn
```

Tests cover: CPU conv2d, CPU layers (instance norm, global avg pool, global dual pool, residual), CPU integration (batch norm, cost func), GPU basic/exact.

## Gotchas

- NCHW layout (channels-first), not NHWC
- `config.inputShape` is `{C, H, W}` — order matters
- Sliding strategies: VALID (no padding), SAME (preserves dims at stride=1), FULL (max padding)
- CNN layers are configured via `config.layersConfig.cnnLayers` (vector of `LayerType` + config pairs)
