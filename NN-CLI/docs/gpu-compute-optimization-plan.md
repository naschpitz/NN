# GPU Compute Optimization: Findings & Plan

## Summary

The `gpu_compute` phase dominates training time because **every sample in a batch triggers a
separate `core->run()` call**, each launching 25–40+ OpenCL kernels with a synchronous
`enqueueBarrier()` after every single kernel. For a batch of 64, this means **1,600–2,500+
kernel launches per batch on a single GPU**, each incurring driver dispatch overhead and a
full pipeline barrier. This is fundamentally why Python NN-CLI (PyTorch) is ~2× faster: it
uses a single cuBLAS GEMM call per layer for the entire batch.

---

## 1. Root Cause: Per-Sample Kernel Launching

### What happens inside one `core->run()` (fast path, no BatchNorm)

```
CNN forward:     im2col + gemm + relu + pool ...     ~3-6  kernels per conv layer
Bridge:          copy_cnn_to_ann                      1     kernel
ANN forward:     calculate_zs + calculate_actvs       ~4-6  kernels (for 2 dense layers)
ANN backward:    dCost_dActv_last + dCost_dBias       ~7-8  kernels (for 2 dense layers)
                 + dCost_dWeight + dCost_dActv
Bridge reverse:  1 kernel
CNN backward:    pool_bwd + relu_bwd + gemm_dW        ~6-8  kernels per conv layer
                 + gemm_dInput + col2im
Accumulate CNN:  accumulate filters + biases          ~2    kernels per conv
Accumulate ANN:  accumulate biases + weights          2     kernels
Loss:            calculate_sample_loss                1     kernel
                 ───────────────────────────────────
                 TOTAL per sample:                    ~25-40 kernels
```

For `batchSize=64`, this is **64 × 25–40 = ~1,600–2,500+ kernel enqueues per batch**.

### Where the overhead comes from

1. **Driver dispatch overhead**: Each `clEnqueueNDRangeKernel` is a system call into the
   GPU driver. 2,000+ calls per batch adds up quickly.

2. **Per-kernel barriers** (`OCLW_CU.cpp:267`): `enqueueBarrier()` is inserted after EVERY
   kernel, forcing the GPU to drain the pipeline before the next kernel starts. Many of
   these barriers are unnecessary — kernels that operate on disjoint buffers could run
   concurrently.

3. **Synchronous transfers** (`OCLW_Core.hpp:130-161`): Every `writeBuffer()` and
   `readBuffer()` calls `waitFinish()` on all compute units, blocking the host thread.
   Additionally, the fast path reads `accum_loss` from GPU after every sample for the
   progress callback (`CNN_CoreGPUWorker.cpp:191`).

4. **No batching**: Kernels operate on vectors of size `numNeurons` or `numWeights` for
   ONE sample. They could operate on `batchSize * numNeurons` if the buffer layout
   supported it, cutting kernel launches by a factor of `batchSize`.

---

## 2. Why Python NN-CLI (PyTorch) Is 2× Faster

| Aspect | NN-CLI C++ (current) | PyTorch |
|--------|---------------------|---------|
| Per-batch kernel launches | ~2,000 (1 per sample per operation) | ~20–30 (1 per layer) |
| Matrix multiply | Single-sample GEMM (1×N) | Batched cuBLAS GEMM (B×N) |
| Kernel dispatch | System call per kernel | Single call per batched kernel |
| Data transfers | Sync with waitFinish each time | Async CUDA streams, overlapping |
| GPU occupancy | Low (small problem per launch) | High (batch fills GPU) |

PyTorch's `torch.nn.Linear.forward(input)` processes the entire `(batchSize × features)`
matrix in a single highly-optimized cuBLAS call. NN-CLI processes one `(1 × features)`
row at a time through hand-written OpenCL kernels.

---

## 3. Current Timing Instrumentation: What It Measures

The timing profiler (`NN-CLI_TrainingProfiler.cpp`) uses a callback from the CNN library
with these phases:

```
Orchestrator thread:
  DataFetch     ─── sampleProvider() call (includes image I/O, decode, augmentation, prefetch)
  GpuTrain      ─── fan-out of trainSubset() across GPUs (wall clock of parallel GPUs)
  GradMerge     ─── multi-GPU gradient merge on CPU
  WeightUpdate  ─── per-GPU weight update kernels (one core->run())
  KernelRestore ─── per-GPU training-kernel restore after update

Per-GPU worker threads:
  H2DUpload     ─── writeBuffer(input) + writeBuffer(output) + writeBuffer(dropoutMask) per sample
  GpuCompute    ─── core->run() — the ENTIRE fused forward+backward+accumulate+loss pipeline
```

### What's missing

- **No per-kernel breakdown**: `GpuCompute` is one number. We can't tell if compute is
  dominated by forward vs backward, CNN vs ANN, or dense vs convolution.
- **No GPU-internal profiling**: All timestamps are host-side CPU `steady_clock` readings.
  OpenCL has built-in GPU profiling (`CL_QUEUE_PROFILING_ENABLE`) that gives actual GPU
  execution time per kernel (the code already exists in `OCLW_CU.cpp:272-283` but is
  disabled).
- **No launch-overhead vs compute separation**: `GpuCompute` mixes kernel launch time
  (enqueue + barrier) with actual GPU execution. The host-side clock includes both.
- **Augmentation hidden in DataFetch**: GPU augmentation time (upload + kernel chain +
  download) is buried inside `DataFetch`, making it impossible to attribute separately.
- **No per-layer breakdown**: We can't profile individual CNN layers or ANN dense layers
  to find hot spots.

---

## 4. Plan: Phased Approach

### Phase 1 — Enable OpenCL Profiling + Collect Per-Kernel Execution Times (no code changes to training logic)

**Goal**: Measure actual GPU kernel execution time (on-device, not host-side) for every
kernel in the pipeline.

**Implementation**:

1. Add `"profiling": true` to the `ComputeUnit` constructor call or add a setter.
   The profiling infrastructure already exists in `OCLW_CU.cpp:272-283` — it just needs
   to be enabled.

2. After each `core->run()`, collect the per-kernel aggregate times from
   `ComputeUnit::kernelTotalTime` and `kernelCallCount`, clear them, and report them
   back through a new callback or log.

3. Add a new `TimingPhase::GpuComputeDetail` or a separate reporting channel that
   exposes per-kernel-type breakdown (e.g., `"calculate_zs": 45.2ms total`).

**Expected outcome**: We'll know exactly what percentage of GPU time goes to:
- `calculate_zs` vs `calculate_actvs` vs `calculate_dCost_dWeight` etc.
- CNN kernels (im2col, gemm, pool, relu) vs ANN kernels
- Forward vs backward
- Actual GPU execution vs host-side overhead (by comparing host-side `GpuCompute` ms
  against the sum of GPU-profiled kernel times)

**Effort**: ~50 lines of code across `OCLW_Core`, `OCLW_CU`, `CNN_CoreGPUWorker`, and
`NN-CLI_TrainingProfiler`.

---

### Phase 2 — Add Sub-Phase Timing Within GpuCompute (host-side)

**Goal**: Separate host-side GpuCompute wall clock into meaningful sub-phases.

**Implementation**:

Add finer-grained `TimingPhase` values:
- `CnnForward` — CNN propagate kernels
- `BridgeCopy` — copy_cnn_to_ann
- `AnnForward` — ANN propagate kernels
- `AnnBackward` — ANN backprop kernels
- `BridgeReverse` — reverse bridge copy
- `CnnBackward` — CNN backprop kernels
- `CNNAccumulate` — CNN gradient accumulation
- `ANNAccumulate` — ANN gradient accumulation
- `LossCompute` — calculate_sample_loss

Emit these from within `CoreGPUWorker::trainSubset()` around the relevant `core->run()`
calls. The fast path currently has a single `core->run()` with all kernels batched — for
sub-phase timing we would need to **split** that into multiple `run()` calls (e.g., one
for forward, one for backward). This would add overhead but is acceptable for a profiling
build (gated behind a `--profile-gpu` flag).

**Alternative (less intrusive)**: Don't split the single `core->run()`. Instead, use
OpenCL event profiling timestamps to attribute GPU time to logical sub-phases. Tag each
kernel with a "phase group" and sum their GPU-profiled times by group.

**Expected outcome**: We'll see whether forward or backward dominates, whether CNN or ANN
dominates, and which specific layer types are the bottleneck.

**Effort**: ~100 lines in `CNN_CoreGPUWorker.cpp` and timing infra.

---

### Phase 3 — Match Potential Optimizations to Measurements

Based on Phase 1+2 data, the optimization strategy will be data-driven:

#### Optimization A — Remove or reduce per-kernel barriers

**Current**: `OCLW_CU.cpp:267` inserts `enqueueBarrier()` after every kernel.

**Fix**: Only insert barriers when there's a data dependency (same buffer written then
read). The `GPUKernelBuilder` knows the buffer dependency graph — it can mark which
kernels need a barrier after them.

**Expected gain**: 10–30% reduction in kernel launch overhead. Kernels on disjoint
buffers (e.g., CNN accumulate and ANN accumulate) can overlap.

#### Optimization B — Fuse kernels

**Current**: Each layer has separate `calculate_zs` and `calculate_actvs` kernels.

**Fix**: Apply the activation function at the end of the zs reduction kernel, eliminating
one kernel launch per layer. Can also fuse `calculate_dCost_dBias` + `calculate_dCost_dWeight`
for the same layer. For Conv layers, fuse im2col + gemm into a single kernel.

**Expected gain**: 15–25% fewer kernels, proportional reduction in launch overhead.

#### Optimization C — PER-BATCH KERNEL BATCHING (highest impact)

**Current**: Each sample runs independently through the kernel queue.

**Fix**: The BatchNorm path already demonstrates per-batch processing — all N samples
coexist in a unified activation buffer and are processed in the same `core->run()`. Extend
this pattern to the non-BatchNorm case:

1. Allocate activation buffers sized for `N × totalActvSize` instead of `1 × totalActvSize`
2. Upload all N samples' inputs at once
3. Run a single `core->run()` per logical phase (forward/backward) that processes all N
   samples through batched kernels

**This requires modifying every GPU kernel** to take the batch size and sample stride as
arguments, and to use the sample index to compute buffer offsets:
```opencl
// Current (single-sample):
uint neuronIdx = get_global_id(0);
float z = bias[neuronIdx];
for (uint w = 0; w < prevNumNeurons; w++)
    z += weight[w * numNeurons + neuronIdx] * actv[w];

// Batched (N samples):
uint neuronIdx = get_global_id(0) % numNeurons;
uint sampleIdx = get_global_id(0) / numNeurons;  // or separate dimension
float z = bias[neuronIdx];
uint weightRow = sampleIdx * prevNumNeurons;  // actvs offset per sample
for (uint w = 0; w < prevNumNeurons; w++)
    z += weight[w * numNeurons + neuronIdx] * actv[weightRow + w];
```

**Expected gain**: 50–80% reduction in `gpu_compute` time. Kernel launches drop from
~2,000/batch to ~30–40/batch. Also improves GPU occupancy since each kernel processes
much more data.

**Effort**: Significant. Every kernel in CNN and ANN needs a variant or parameterization
for batch mode. This is the Torch-porting effort retried but with a different strategy —
reuse the existing kernel structure but add a batch dimension, rather than rewrite from
scratch.

#### Optimization D — Async transfers + multi-buffering

**Current**: All buffer writes are synchronous with `waitFinish()`.

**Fix**: Use `clEnqueueWriteBuffer(..., CL_FALSE, ...)` with event-based synchronization.
Double-buffer activations: while GPU processes batch N, upload batch N+1.

**Expected gain**: Overlap H2D transfers with compute, potentially hiding 5–15% of time.

#### Optimization E — Reduce progress callback reads

**Current**: `accum_loss` is read from GPU after every sample for the progress callback
(`CNN_CoreGPUWorker.cpp:191`).

**Fix**: Read only once per batch, or compute running loss on the host side from the
per-sample delta that's already being computed.

**Expected gain**: Eliminates N-1 synchronous GPU→CPU transfers per batch.

---

### Phase 4 — Augmentation Profiling

**Goal**: Separate augmentation timing from `DataFetch`.

**Implementation**: Add a new `TimingPhase::Augmentation` emitted around the
`gpuAugmenterPool->augment()` call in `DataLoader::loadBatch()`.

**Expected outcome**: Know how much of `DataFetch` is augmentation vs actual I/O.

---

## 5. End-to-End Optimized Pipeline (Target State)

```
Per batch (no BatchNorm, batchSize=64, single GPU):

BEFORE (current):
  DataFetch:     load 64 images + augment on GPU + wait for prefetch
  H2DUpload:     64 × writeBuffer(input) + 64 × writeBuffer(output) + dropout masks
                 = 128+ synchronous transfers
  GpuCompute:    64 × core->run()
                 = 64 × (~30 kernels with barriers) = ~1,920 kernel enqueues
  GradMerge:     sum gradients (single GPU = no-op essentially)
  WeightUpdate:  1 × core->run() (update kernels)
  KernelRestore: restore training kernels
  ─────────────────────────────────────────────────────────────────
  ~2,000+ kernel launches per batch

AFTER (target):
  DataFetch:     async load 64 images + async GPU augment (overlapped)
  H2DUpload:     1 × writeBuffer(all 64 inputs) + 1 × writeBuffer(all 64 outputs)
  GpuForward:    1 × core->run() (batched CNN forward + bridge + ANN forward)
                 = ~10 kernels (im2col+gemm+relu+pool + bridge + zs+actvs)
  H2DUpload:     async dropout masks
  GpuBackward:   1 × core->run() (batched ANN backward + bridge reverse + CNN backward
                 + accumulate CNN + accumulate ANN + loss)
                 = ~15 kernels
  GradMerge:     sum gradients (single GPU = no-op)
  WeightUpdate:  1 × core->run()
  ─────────────────────────────────────────────────────────────────
  ~28 kernel launches per batch (70× reduction)
```

---

## 6. Recommended First Steps

1. **Enable OpenCL profiling** (`CL_QUEUE_PROFILING_ENABLE`) and log per-kernel-type
   GPU times. This requires minimal code changes (~20 lines) and gives immediate insight.

2. **Add augmentation timing** as a separate profiler phase to expose how much of
   `DataFetch` is actually augmentation GPU time.

3. **Remove per-kernel barriers** where safe. This is a quick win with low risk.
   Audit the kernel dependency graph and only insert barriers on actual read-after-write
   hazards within the same buffer.

4. **Start kernel batching with the simplest case first**: ANN dense layers (pure
   matmul — easiest to batch). Measure the gain, then extend to CNN layers.

5. **Benchmark Python NN-CLI** running the exact same model/config to get a precise
   performance gap measurement (not just "~2×"), then use the profiling data to
   target the specific operations where the gap is widest.

---

## 7. File Map

| File | Role in this analysis |
|------|----------------------|
| `NN-CLI/NN-CLI_TrainingProfiler.{hpp,cpp}` | Host-side timing consumer. Coarse phases only. |
| `NN-CLI/NN-CLI_CNNRunner.cpp` | Wires TimingCallback from CNN to profiler. |
| `NN-CLI/NN-CLI_DataLoader.cpp` | Batch loading + GPU augmentation call (inside DataFetch). |
| `NN-CLI/NN-CLI_GpuAugmenter.{hpp,cpp}` | GPU image augmentation (per-batch, ~1-16 kernels). |
| `extern/CNN/CNN_CoreGPU.cpp` | Training orchestrator: epoch/batch loop, emits DataFetch/GpuTrain/GradMerge/WeightUpdate/KernelRestore. |
| `extern/CNN/CNN_CoreGPUWorker.cpp` | Per-GPU worker: trainSubset() emits H2DUpload/GpuCompute. Fast path = per-sample core->run(). BN path = per-batch segments. |
| `extern/CNN/CNN_GPUKernelBuilder.cpp` | Builds all OpenCL kernel queues (forward, backward, accumulate, update). ~1,469 lines. |
| `extern/CNN/CNN_GPUBufferManager.{hpp,cpp}` | Buffer allocation and offset computation. |
| `extern/ANN/ANN_CoreGPUWorker.cpp` | ANN per-sample training loop (called by CNN worker for ANN portion). |
| `extern/ANN/ANN_GPUKernelBuilder.cpp` | ANN kernel queue builder (propagate, backpropagate, accumulate, update). |
| `extern/ANN/ANN_GPUBufferManager.{hpp,cpp}` | ANN buffer allocation. |
| `extern/OpenCLWrapper/OCLW_CU.cpp` | Kernel enqueue + barriers + optional profiling (lines 223-286). |
| `extern/OpenCLWrapper/OCLW_Core.hpp` | Synchronous writeBuffer/readBuffer with waitFinish after every transfer. |
| `extern/ANN/ANN_CoreGPU.cpp` | Multi-GPU orchestrator, gradient merge on host. |
