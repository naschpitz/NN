# Adam Optimizer Implementation Plan

## 1. Overview

Add Adam optimizer support to the ANN and CNN libraries, with JSON configuration parsing in NN-CLI. The current SGD-only update logic will be extended to support both SGD and Adam, selectable via configuration. Default remains SGD for backward compatibility.

**Adam algorithm per parameter θ:**
```
g = gradient / numSamples
m = β1·m + (1-β1)·g
v = β2·v + (1-β2)·g²
m̂ = m / (1-β1^t)
v̂ = v / (1-β2^t)
θ = θ - lr·m̂ / (√v̂ + ε)
```

**Success criteria:** Existing SGD behavior unchanged when no optimizer is specified. Adam selectable via `"optimizer": {"type": "adam", "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8}` in JSON config. Works on both CPU and GPU paths for both ANN and CNN.

## 2. Prerequisites
- No new dependencies required
- No migrations needed
- Changes are purely additive to existing structs/classes

## 3. Implementation Steps

### Step 1: Add OptimizerConfig to ANN TrainingConfig

**File: `extern/CNN/extern/ANN/ANN_TrainingConfig.hpp`**

Add an `OptimizerType` enum and `OptimizerConfig` struct, then embed in `TrainingConfig`:

```cpp
enum class OptimizerType { SGD, ADAM };

template <typename T>
struct OptimizerConfig {
    OptimizerType type = OptimizerType::SGD;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
};

template <typename T>
struct TrainingConfig {
    ulong numEpochs = 0;
    float learningRate = 0.01f;
    ulong batchSize = 64;
    bool shuffleSamples = true;
    float dropoutRate = 0.0f;
    OptimizerConfig<T> optimizer; // NEW — defaults to SGD
};
```

### Step 2: Add OptimizerConfig to CNN TrainingConfig

**File: `extern/CNN/CNN_TrainingConfig.hpp`**

Mirror the same structure. Since CNN includes ANN headers, we can reference ANN's enum or duplicate it in the CNN namespace for consistency with the existing pattern (each library has its own namespace-local copies of types like `CostFunctionType`).

```cpp
enum class OptimizerType { SGD, ADAM };

template <typename T>
struct OptimizerConfig {
    OptimizerType type = OptimizerType::SGD;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
};

template <typename T>
struct TrainingConfig {
    // ... existing fields ...
    OptimizerConfig<T> optimizer; // NEW
};
```

### Step 3: Add Adam state buffers to ANN CoreCPU

**File: `extern/CNN/extern/ANN/ANN_CoreCPU.hpp`** — Add member variables:
```cpp
// Adam optimizer state (first and second moment estimates)
Tensor3D<T> adam_m_weights;  // first moment for weights
Tensor2D<T> adam_m_biases;   // first moment for biases
Tensor3D<T> adam_v_weights;  // second moment for weights
Tensor2D<T> adam_v_biases;   // second moment for biases
ulong adam_t = 0;            // timestep counter
```

**File: `extern/CNN/extern/ANN/ANN_CoreCPU.cpp`** — Changes:
1. In `allocateGlobalAccumulators()`: also allocate `adam_m_*` and `adam_v_*` buffers (same shape as accumulators), initialized to zero, but only if `optimizer.type == ADAM`.
2. In `update(ulong numSamples)`: branch on `this->trainingConfig.optimizer.type`:
   - **SGD path**: existing code (unchanged)
   - **ADAM path**: increment `adam_t`, then for each weight/bias:
     ```cpp
     T g = accumWeights[l][j][k] / numSamples;
     adam_m_weights[l][j][k] = beta1 * adam_m_weights[l][j][k] + (1 - beta1) * g;
     adam_v_weights[l][j][k] = beta2 * adam_v_weights[l][j][k] + (1 - beta2) * g * g;
     T mHat = adam_m_weights[l][j][k] / (1 - std::pow(beta1, adam_t));
     T vHat = adam_v_weights[l][j][k] / (1 - std::pow(beta2, adam_t));
     this->parameters.weights[l][j][k] -= lr * mHat / (std::sqrt(vHat) + epsilon);
     ```
     Same pattern for biases.

### Step 4: Add Adam state buffers to CNN CoreCPU

**File: `extern/CNN/CNN_CoreCPU.hpp`** — Add member variables:
```cpp
// Adam state for CNN conv parameters
std::vector<std::vector<T>> adam_m_convFilters;
std::vector<std::vector<T>> adam_m_convBiases;
std::vector<std::vector<T>> adam_v_convFilters;
std::vector<std::vector<T>> adam_v_convBiases;
ulong adam_t = 0;
```

**File: `extern/CNN/CNN_CoreCPU.cpp`** — Changes:
1. In constructor: allocate Adam state vectors (same shape as `accumDConvFilters`/`accumDConvBiases`) if optimizer is ADAM.
2. In `updateCNNParameters(ulong numSamples)`: branch on optimizer type, apply Adam update for conv filters/biases.

### Step 5: Propagate optimizer config from CNN to ANN sub-core

**File: `extern/CNN/CNN_CoreCPUWorker.cpp`** — In `buildANNConfig()`, add:
```cpp
annConfig.trainingConfig.optimizer.type = static_cast<ANN::OptimizerType>(cnnConfig.trainingConfig.optimizer.type);
annConfig.trainingConfig.optimizer.beta1 = cnnConfig.trainingConfig.optimizer.beta1;
annConfig.trainingConfig.optimizer.beta2 = cnnConfig.trainingConfig.optimizer.beta2;
annConfig.trainingConfig.optimizer.epsilon = cnnConfig.trainingConfig.optimizer.epsilon;
```


### Step 6: Add Adam GPU kernels to ANN OpenCL

**File: `extern/CNN/extern/ANN/opencl/Kernels.cpp.cl`** — Add two new kernels after the existing `update_weights`/`update_biases`:

```opencl
kernel void update_biases_adam(
    global TYPE* biases,
    global TYPE* accum_dCost_dBiases,
    global TYPE* adam_m_biases,
    global TYPE* adam_v_biases,
    ulong numSamples,
    float learningRate,
    float beta1,
    float beta2,
    float epsilon,
    ulong timestep,
    ulong size
) {
  size_t idx = get_global_id(0);
  if (idx >= size) return;

  TYPE g = accum_dCost_dBiases[idx] / (TYPE)numSamples;
  TYPE m = beta1 * adam_m_biases[idx] + (1.0f - beta1) * g;
  TYPE v = beta2 * adam_v_biases[idx] + (1.0f - beta2) * g * g;
  adam_m_biases[idx] = m;
  adam_v_biases[idx] = v;

  TYPE mHat = m / (1.0f - pown((TYPE)beta1, (int)timestep));
  TYPE vHat = v / (1.0f - pown((TYPE)beta2, (int)timestep));
  biases[idx] -= learningRate * mHat / (sqrt(vHat) + (TYPE)epsilon);
}

kernel void update_weights_adam(
    global TYPE* weights,
    global TYPE* accum_dCost_dWeights,
    global TYPE* adam_m_weights,
    global TYPE* adam_v_weights,
    ulong numSamples,
    float learningRate,
    float beta1,
    float beta2,
    float epsilon,
    ulong timestep,
    ulong size
) {
  size_t idx = get_global_id(0);
  if (idx >= size) return;

  TYPE g = accum_dCost_dWeights[idx] / (TYPE)numSamples;
  TYPE m = beta1 * adam_m_weights[idx] + (1.0f - beta1) * g;
  TYPE v = beta2 * adam_v_weights[idx] + (1.0f - beta2) * g * g;
  adam_m_weights[idx] = m;
  adam_v_weights[idx] = v;

  TYPE mHat = m / (1.0f - pown((TYPE)beta1, (int)timestep));
  TYPE vHat = v / (1.0f - pown((TYPE)beta2, (int)timestep));
  weights[idx] -= learningRate * mHat / (sqrt(vHat) + (TYPE)epsilon);
}
```

### Step 7: Add Adam GPU kernel to CNN OpenCL

**File: `extern/CNN/opencl/CNN_Kernels.cpp.cl`** — Add after existing `update_parameters`:

```opencl
kernel void update_parameters_adam(
    global TYPE* params,
    global TYPE* accum,
    global TYPE* adam_m,
    global TYPE* adam_v,
    ulong offset,
    ulong size,
    ulong numSamples,
    float learningRate,
    float beta1,
    float beta2,
    float epsilon,
    ulong timestep
) {
  size_t gid = get_global_id(0);
  if (gid >= size) return;

  ulong i = offset + gid;
  TYPE g = accum[i] / (TYPE)numSamples;
  TYPE m = beta1 * adam_m[i] + (1.0f - beta1) * g;
  TYPE v = beta2 * adam_v[i] + (1.0f - beta2) * g * g;
  adam_m[i] = m;
  adam_v[i] = v;

  TYPE mHat = m / (1.0f - pown((TYPE)beta1, (int)timestep));
  TYPE vHat = v / (1.0f - pown((TYPE)beta2, (int)timestep));
  params[i] -= learningRate * mHat / (sqrt(vHat) + (TYPE)epsilon);
}
```

### Step 8: Add Adam GPU buffers and kernel setup to ANN CoreGPUWorker

**File: `extern/CNN/extern/ANN/ANN_CoreGPUWorker.hpp`** — Add member:
```cpp
ulong adam_t = 0;
```

**File: `extern/CNN/extern/ANN/ANN_CoreGPUWorker.cpp`** — Changes:

1. In `allocateBuffers()`: if optimizer is ADAM, allocate four additional GPU buffers:
   ```cpp
   if (this->trainingConfig.optimizer.type == OptimizerType::ADAM) {
     this->core->template allocateBuffer<T>("adam_m_weights", totalNumWeights);
     this->core->template allocateBuffer<T>("adam_v_weights", totalNumWeights);
     this->core->template allocateBuffer<T>("adam_m_biases", totalNumBiases);
     this->core->template allocateBuffer<T>("adam_v_biases", totalNumBiases);
   }
   ```

2. In `addUpdateKernels(ulong numSamples)`: branch on optimizer type:
   - **SGD**: existing code (unchanged)
   - **ADAM**: increment `adam_t`, use `update_biases_adam` and `update_weights_adam` kernels instead, passing the adam buffers and hyperparameters as arguments.

### Step 9: Add Adam GPU buffers and kernel setup to CNN CoreGPUWorker

**File: `extern/CNN/CNN_CoreGPUWorker.hpp`** — Add member:
```cpp
ulong adam_t = 0;
```

**File: `extern/CNN/CNN_CoreGPUWorker.cpp`** — Changes:

1. In `allocateBuffers()`: if optimizer is ADAM, allocate:
   ```cpp
   if (this->coreConfig.trainingConfig.optimizer.type == OptimizerType::ADAM) {
     if (this->totalFilterSize > 0) {
       this->core->template allocateBuffer<T>("cnn_adam_m_filters", this->totalFilterSize);
       this->core->template allocateBuffer<T>("cnn_adam_v_filters", this->totalFilterSize);
     }
     if (this->totalBiasSize > 0) {
       this->core->template allocateBuffer<T>("cnn_adam_m_biases", this->totalBiasSize);
       this->core->template allocateBuffer<T>("cnn_adam_v_biases", this->totalBiasSize);
     }
   }
   ```

2. In `addCNNUpdateKernels(ulong numSamples)`: branch on optimizer type:
   - **SGD**: existing code (unchanged)
   - **ADAM**: increment `adam_t`, use `update_parameters_adam` kernel, passing adam buffers and hyperparameters.

3. In `setupUpdateKernels(ulong numSamples)`: no structural change needed — it already calls `addCNNUpdateKernels` + `annGPUWorker->addUpdateKernels`, both of which will internally branch.

### Step 10: Parse optimizer config from JSON in NN-CLI

**File: `NN-CLI_Loader.cpp`** — In the `loadANNConfig` function, inside the `if (json.contains("trainingConfig"))` block, add after existing fields:

```cpp
if (tc.contains("optimizer")) {
  const auto& opt = tc.at("optimizer");
  std::string optType = opt.at("type").get<std::string>();

  if (optType == "adam") {
    coreConfig.trainingConfig.optimizer.type = ANN::OptimizerType::ADAM;

    if (opt.contains("beta1"))
      coreConfig.trainingConfig.optimizer.beta1 = opt.at("beta1").get<float>();
    if (opt.contains("beta2"))
      coreConfig.trainingConfig.optimizer.beta2 = opt.at("beta2").get<float>();
    if (opt.contains("epsilon"))
      coreConfig.trainingConfig.optimizer.epsilon = opt.at("epsilon").get<float>();
  }
  // else: default SGD (already set)
}
```

Same pattern in `loadCNNConfig`, using `CNN::OptimizerType::ADAM`.

### Step 11: JSON config example

After implementation, a config using Adam would look like:

```json
{
  "trainingConfig": {
    "numEpochs": 30,
    "learningRate": 0.001,
    "batchSize": 512,
    "optimizer": {
      "type": "adam",
      "beta1": 0.9,
      "beta2": 0.999,
      "epsilon": 1e-8
    }
  }
}
```

If `"optimizer"` is omitted, behavior defaults to SGD (backward compatible).

## 4. File Changes Summary

### Files to Modify (no new files created):

| # | File | Changes |
|---|------|---------|
| 1 | `extern/CNN/extern/ANN/ANN_TrainingConfig.hpp` | Add `OptimizerType` enum, `OptimizerConfig<T>` struct, add `optimizer` field to `TrainingConfig` |
| 2 | `extern/CNN/CNN_TrainingConfig.hpp` | Same as above for CNN namespace |
| 3 | `extern/CNN/extern/ANN/ANN_CoreCPU.hpp` | Add Adam state members (`adam_m_*`, `adam_v_*`, `adam_t`) |
| 4 | `extern/CNN/extern/ANN/ANN_CoreCPU.cpp` | Allocate Adam state in `allocateGlobalAccumulators()`; branch in `update()` for Adam vs SGD |
| 5 | `extern/CNN/CNN_CoreCPU.hpp` | Add Adam state members for conv params |
| 6 | `extern/CNN/CNN_CoreCPU.cpp` | Allocate Adam state in constructor; branch in `updateCNNParameters()` |
| 7 | `extern/CNN/CNN_CoreCPUWorker.cpp` | Propagate optimizer config in `buildANNConfig()` |
| 8 | `extern/CNN/extern/ANN/opencl/Kernels.cpp.cl` | Add `update_biases_adam` and `update_weights_adam` kernels |
| 9 | `extern/CNN/opencl/CNN_Kernels.cpp.cl` | Add `update_parameters_adam` kernel |
| 10 | `extern/CNN/extern/ANN/ANN_CoreGPUWorker.hpp` | Add `adam_t` member |
| 11 | `extern/CNN/extern/ANN/ANN_CoreGPUWorker.cpp` | Allocate Adam GPU buffers in `allocateBuffers()`; branch in `addUpdateKernels()` |
| 12 | `extern/CNN/CNN_CoreGPUWorker.hpp` | Add `adam_t` member |
| 13 | `extern/CNN/CNN_CoreGPUWorker.cpp` | Allocate Adam GPU buffers in `allocateBuffers()`; branch in `addCNNUpdateKernels()`; propagate optimizer config in `buildANNWorker()` |
| 14 | `NN-CLI_Loader.cpp` | Parse `"optimizer"` object from JSON in both `loadANNConfig()` and `loadCNNConfig()` |

### Files NOT modified (no changes needed):
- `ANN_CoreCPUWorker.hpp/.cpp` — Workers only compute/accumulate gradients; they don't do parameter updates.
- `ANN_Core.hpp/.cpp` — Base class stores `trainingConfig` which will automatically include the new `optimizer` field.
- `ANN_CoreGPU.hpp/.cpp` — Delegates to `CoreGPUWorker`; no direct update logic.
- `CNN_CoreGPU.hpp/.cpp` — Delegates to `CoreGPUWorker`; no direct update logic.
- `CNN_CoreGPUWorker.hpp` — Only needs `adam_t` (already listed).

## 5. Testing Strategy

**Unit tests:**
- Verify that existing tests pass with no config changes (SGD default path).
- Add a test with `optimizer.type = ADAM` for ANN CPU training on the XOR-like dataset (similar to existing `ann_train_config.json` tests).
- Add a test with `optimizer.type = ADAM` for CNN CPU training.
- Verify Adam converges (loss decreases over epochs) — does not need to match SGD exactly.

**GPU tests (if GPU available):**
- Run existing GPU tests to verify SGD still works.
- Add GPU Adam test to verify convergence.

**Manual testing:**
- Create a JSON config with `"optimizer": {"type": "adam"}` and run NN-CLI training.
- Verify that omitting `"optimizer"` still works (SGD).
- Verify that `"optimizer": {"type": "sgd"}` works explicitly.

## 6. Rollback Plan

All changes are additive:
- The `OptimizerConfig` struct defaults to SGD.
- JSON parsing only activates Adam if `"optimizer"` key is present.
- Reverting = removing the new fields/methods/kernels.
- No data migration needed — model save/load format is unaffected (Adam state is transient, not serialized).

## 7. Estimated Effort

- **Complexity:** Medium
- **Estimated time:** ~4-6 hours of implementation
- **Breakdown:**
  - TrainingConfig changes + JSON parsing: ~30 min
  - ANN CPU Adam implementation: ~1 hour
  - CNN CPU Adam implementation: ~1 hour
  - ANN GPU kernels + buffer allocation: ~1 hour
  - CNN GPU kernels + buffer allocation: ~1 hour
  - CNN→ANN config propagation: ~15 min
  - Testing: ~1 hour

## 8. Key Design Decisions

1. **Adam state is per-Core, not per-Worker.** Workers compute gradients; the Core (or CoreGPUWorker) owns the update logic and Adam state. This matches the existing architecture where `update()` is called on the Core after merging worker accumulators.

2. **Timestep `t` increments once per `update()` call** (i.e., once per mini-batch). This is standard Adam behavior.

3. **Adam state is NOT serialized** in model saves. When resuming training, Adam moments restart from zero. This is a common simplification; adding serialization can be a follow-up.

4. **GPU Adam kernels use `pown()` for β^t** computation. For very large `t`, this is numerically stable since β<1. Alternative: precompute `1-β1^t` on CPU and pass as kernel argument (slightly more efficient, avoids GPU power computation). **Recommendation: precompute on CPU** and pass `bias_correction1 = 1-β1^t` and `bias_correction2 = 1-β2^t` as float arguments instead of `timestep`. This avoids potential precision issues with `pown()` for large t values.
