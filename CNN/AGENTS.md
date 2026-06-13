# CNN — Convolutional Neural Network Library

C++ CNN library (conv/pool/norm/residual layers) with CPU and GPU (OpenCL) backends. Delegates dense layers to ANN.

## Build

```bash
./build.sh            # from repo root (all components)
./CNN/build.sh        # standalone
```

Builds into `CNN/build/`. Test binary: `test_cnn`.

## Architecture

- `Core` (abstract) → `CoreCPU` / `CoreGPU`
- `CoreCPUWorker` / `CoreGPUWorker` handle per-batch parallelism
- `CNN_GPUKernelBuilder` compiles OpenCL kernels from `opencl/*.cl`
- `CNN_CoreGPUWorkerConfig` handles GPU kernel configuration
- Dense layers (after flatten) are delegated to `ANN::Core<float>` per-sample

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
