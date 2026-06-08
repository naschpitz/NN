# Micro-Batch GPU Processing — Implementation Plan

## Goal

Process `microBatchSize=64` samples concurrently in each GPU kernel launch, instead of
the current fast-path behavior of processing one sample at a time (microBatchSize=1
hardcoded). This eliminates the per-sample `core->run()` loop and reduces host-side
round-trips by a factor of 64.

## Target Config: ISIC-MILK10k train-app-11

| Parameter | Value |
|-----------|-------|
| CNN layers | 41 (residual blocks, InstanceNorm, GlobalDualPool) |
| Dense layers | 2 (128 neurons, 2 output) |
| batchSize | 4096 |
| microBatchSize | 64 (ignored by code) |
| InstanceNorm | yes (per-sample — no BN path triggered) |
| BatchNorm | **no** — fast path is used |
| Current speed | **125-126 img/s** |
| Config file | `examples/ISIC-MILK10k/train-app-11/isic_cnn_config.json` |
| Train samples | `examples/ISIC-MILK10k/train-app-11/samples.json` (469,688) |
| Test samples | `examples/ISIC-MILK10k/train-app-11/samples_test.json` (7,937) |

### Test command

```bash
./build_test/NN-CLI \
  --config examples/ISIC-MILK10k/train-app-11/isic_cnn_config.json \
  --mode train \
  --samples examples/ISIC-MILK10k/train-app-11/samples_test.json \
  --output /tmp/output-test/trained.json
```

The `--output` flag redirects the saved model to avoid overwriting the
legitimate training output in `examples/.../output/`.

For faster iteration, override epochs to 1 and use a small subset:

```bash
sed 's/"numEpochs": 100/"numEpochs": 1/' \
  examples/ISIC-MILK10k/train-app-11/isic_cnn_config.json > /tmp/test_1e.json

python3 -c "
import json
with open('examples/ISIC-MILK10k/train-app-11/samples_test.json') as f: d = json.load(f)
d['samples'] = d['samples'][:64]
with open('/tmp/test_64s.json','w') as f: json.dump(d,f)
"

./build_test/NN-CLI --config /tmp/test_1e.json --mode train \
  --samples /tmp/test_64s.json --output /tmp/output-test/trained.json
```

---

## Current State

The fast path (no BatchNorm) in `CNN_CoreGPUWorker::trainSubset()` processes each
sample independently:

```
for each sample s in miniBatch:
    writeBuffer(input, offset=0)           ← overwrites same buffer slot
    writeBuffer(expectedOutput, offset=0)
    writeBuffer(dropoutMask)
    core->run()                            ← all CNN+ANN forward+backward kernels for ONE sample
```

All samples share buffer offset 0. The kernel queue is built once with `sampleIdx=0`
and reused for every sample. microBatchSize=64 in the config is never read.

The BatchNorm path already demonstrates per-batch processing: all N samples coexist
in a unified activation buffer at offsets `n * sampleStride`. Kernels reference
`sampleIdx` to compute the correct offset. The forward and backward passes iterate
samples within a single `core->run()` call.

## Approach: Adapt BatchNorm Path Pattern to Fast Path

Do NOT modify GPU kernels. Instead, use the same mechanism the BN path uses:
build N copies of each kernel (one per sample) into a single kernel queue, then
`core->run()` once per phase instead of once per sample.

App-11 has InstanceNorm (per-sample), not BatchNorm (cross-sample). InstanceNorm
is already handled correctly in the fast path — each sample computes its own
mean/variance. No special cross-sample handling needed.

This yields:
- **Kernel launches per batch**: unchanged (~188 × N per micro-batch), but all in one queue
- **`core->run()` calls per batch**: reduced from N to ~4 (forward/backward phases)
- **Memory**: increased by `microBatchSize`× for activation buffers (same as BN path)
- **GPU occupancy improvement**: marginal (kernels still per-sample, just queued together)

A later optimization (kernel-level batching) would get the full speedup by having
each kernel process all N samples internally, but that requires kernel rewrites.

---

## Files to Change

### A. Config Plumbing (NN-CLI side)

**`NN-CLI_Loader.hpp/cpp`** — Loader helpers
- Read `training.microBatchSize` from config JSON
- Store in `TrainingConfig`

**`extern/CNN/CNN_TrainingConfig.hpp`** — Config struct
- Add `ulong microBatchSize = 1` field (default 1 = current behavior)

**`extern/CNN/CNN_Core.hpp`** — Core base class
- Read `trainingConfig.microBatchSize` into `CoreGPUWorkerConfig`

**`extern/CNN/CNN_CoreGPUWorkerConfig.hpp`** — Worker config
- Add `ulong microBatchSize = 1` field

### B. Buffer Allocation

**`extern/CNN/CNN_GPUBufferManager.cpp`** — `allocateBuffers()`
- In the non-BN path, use `microBatchSize` instead of 1 to compute activation
  buffer multiplier. The BN path already uses `batchSize`; extend this to the
  fast path when `microBatchSize > 1`.
- `sampleStride = totalActvSize` (per-sample size, already correct).
- `cnn_actvs` buffer allocated as `microBatchSize * totalActvSize`.

**`extern/CNN/CNN_GPUBufferManager.hpp`** — `totalActvSize`
- Unchanged. `totalActvSize` is sum of per-sample activation sizes.

### C. Kernel Builder

**`extern/CNN/CNN_GPUKernelBuilder.hpp/cpp`**

Add per-phase setup methods that iterate sample indices:

```
setupMicroCnnForwardKernels(count, startIdx)   — addPropagateKernels(n, ...) for n=0..count-1
setupMicroAnForwardKernels(count, startIdx)    — addCopyBridgeKernels(n) + ANN addPropagateKernels
setupMicroAnBackwardKernels(count, startIdx)   — ANN addBackpropagateKernels + addReverseBridgeKernels(n)
                                                 + ANN addAccumulateKernels + loss
setupMicroCnnBackwardKernels(count, startIdx)  — addBackpropagateKernels(n, ...) + addCNNAccumulateKernels(n, ...)
```

The existing `addPropagateKernels(n, ...)`, `addBackpropagateKernels(n, ...)`,
`addCopyBridgeKernels(n)`, `addReverseBridgeKernels(n)` already accept `sampleIdx`.
Each micro-method just loops over `n` and calls the existing functions.

Kernel IDs are unique per sample via the existing `_s{n}_l{layerIdx}` suffix.

### D. Training Loop

**`extern/CNN/CNN_CoreGPUWorker.cpp`** — `trainSubset()`

When `!hasBatchNorm && microBatchSize > 1`:

```cpp
ulong microSize = this->workerConfig.microBatchSize;
ulong numMicroBatches = (N + microSize - 1) / microSize;

for (ulong mb = 0; mb < numMicroBatches; mb++) {
    ulong mbStart = mb * microSize;
    ulong mbEnd = std::min(mbStart + microSize, N);
    ulong mbN = mbEnd - mbStart;

    // 1. Upload all mbN inputs to their sample offsets
    emit(H2DUpload, Begin);
    for (ulong n = 0; n < mbN; n++) {
        writeBuffer("cnn_actvs", input[mbStart+n], n * sampleStride);
    }
    emit(H2DUpload, End);

    // 2. CNN forward: all mbN samples in one core->run()
    emit(GpuCompute, Begin);
    clearKernels();
    setupMicroCnnForwardKernels(mbN);
    core->run();

    // 3. Per-sample: ANN forward + backward (dropout/expected are per-sample)
    for (ulong n = 0; n < mbN; n++) {
        clearKernels();
        addCopyBridgeKernels(n);
        annWorker->kernelBuilder->addPropagateKernels();

        writeBuffer("outputs", expected[mbStart+n], 0);
        if (hasDropout) uploadDropoutMask();

        annWorker->kernelBuilder->addBackpropagateKernels(true);
        addReverseBridgeKernels(n);
        annWorker->kernelBuilder->addAccumulateKernels();
        addLossKernel(n);

        core->run();

        // Progress callback (read accum_loss)
        if (callback) { ... }
    }

    // 4. CNN backward: all mbN samples in one core->run()
    clearKernels();
    setupMicroCnnBackwardKernels(mbN);
    core->run();
    emit(GpuCompute, End);
}
```

### E. Optimized Variant: Merge ANN Steps

Since the ANN portion (2 dense layers with 128 and 2 neurons) is tiny compared
to the 41 CNN layers, we can fold the per-sample ANN loop into a combined
forward+backward run per sample, sharing the CNN forward/backward batched runs:

A simpler split: just two `core->run()` calls per micro-batch:
1. CNN forward (batched, N samples)
2. Per-sample: ANN forward + ANN backward + CNN backward + accumulate (all per sample)

Or even simpler: keep the ANN per-sample but coalesce CNN backward per micro-batch.

---

## Acceptance Criteria

- `microBatchSize=1` produces identical gradients/weights as current code (bit-exact)
- `microBatchSize=64` produces identical gradients as `microBatchSize=1` (different
  order of accumulation, but same final result since gradients are summed)
- No regression in img/s at `microBatchSize=1`
- For `microBatchSize=64`, observe reduction in `gpu_compute` wall clock time
  (fewer `core->run()` calls) and the GPU profiling breakdown shows the same
  per-kernel times but with fewer host round-trips

---

## Performance Measurement

After implementation, compare img/s:

```bash
# Baseline (current code)
./build/NN-CLI --config examples/ISIC-MILK10k/train-app-11/isic_cnn_config.json \
  --mode train --samples examples/ISIC-MILK10k/train-app-11/samples_test.json \
  --output /tmp/output-test/trained.json
# expect ~125 img/s

# With microBatchSize=64
./build_test/NN-CLI --config examples/ISIC-MILK10k/train-app-11/isic_cnn_config.json \
  --mode train --samples examples/ISIC-MILK10k/train-app-11/samples_test.json \
  --output /tmp/output-test/trained.json
# target: measurable improvement from fewer core->run() calls
```

Use the GPU profiling instrumentation to compare:
- `gpu_compute` host-side wall clock (should decrease)
- Per-kernel GPU execution times (should be identical — same work, same kernels)

---

## Memory Impact (app-11)

41 CNN layers with residual blocks + InstanceNorm + GlobalDualPool.
Estimated per-sample activation buffer: ~4-8 MB.

| microBatchSize | Approx extra VRAM |
|---|---|
| 1 (current) | baseline |
| 64 | ~256-512 MB |
| 128 | ~512 MB-1 GB |

GPUs with 8GB+ VRAM handle microBatchSize=128 comfortably.

---

## Implementation Order

1. **Config plumbing** — add `microBatchSize` to TrainingConfig, CoreGPUWorkerConfig
2. **Buffer allocation** — use `microBatchSize` in fast-path buffer sizing
3. **Kernel builder** — add per-phase `setupMicro*` methods
4. **Training loop** — micro-batch loop in `trainSubset()`
5. **Test correctness** — gradient match against CPU path
6. **Benchmark** — measure img/s at microBatchSize=1, 16, 32, 64, 128
