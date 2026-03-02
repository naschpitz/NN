# CoreGPUWorker Refactoring Plan

## 1. Overview

Decompose both `ANN::CoreGPUWorker` (~985 lines .cpp) and `CNN::CoreGPUWorker` (~1137 lines .cpp) into three focused classes each using **composition**:

1. **`GPUKernelBuilder<T>`** — Kernel building blocks + kernel setup orchestration
2. **`GPUBufferManager<T>`** — Buffer allocation, source loading, parameter init/sync, data I/O, offset computation
3. **`CoreGPUWorker<T>`** — Slimmed coordinator/facade with execution logic

**Goals:**
- No file exceeds ~450 lines in .cpp
- Public API from `CoreGPU`'s perspective is **unchanged** (delegation wrappers)
- CNN's ability to call ANN's kernel building methods is preserved
- Shared-core pattern (owned vs external `OpenCLWrapper::Core`) is preserved

**Scope:** Internal decomposition only. No changes to `CoreGPU`, `Worker`, or any callers.

## 2. Architecture

```
CoreGPUWorker<T>  (facade — owns core pointer, orchestrates execution)
  ├── GPUBufferManager<T>  (member, constructed first)
  │     ├── owns: layout data (offsets, sizes)
  │     ├── owns: ANN::CoreGPUWorker* (CNN only)
  │     ├── methods: loadSources, allocateBuffers, initializeParameters,
  │     │            syncParametersFromGPU, readOutput, readInputGradients,
  │     │            readAccumulatedGradients, setAccumulators,
  │     │            generateAndUploadDropoutMask, offset helpers
  │     └── receives: core* (raw, non-owning), config refs, parameters ref
  │
  └── GPUKernelBuilder<T>  (member, constructed second)
        ├── owns: kernel flags, adam_t state
        ├── methods: addPropagateKernels, addBackpropagateKernels,
        │            addAccumulateKernels, addUpdateKernels,
        │            setup*Kernels, invalidateAllKernelFlags
        └── receives: core* (raw), config refs (const), parameters ref (const)
                      For CNN: also receives GPUBufferManager& for layout data + ANN worker access
```

**Lifetime:** Both components are `std::unique_ptr` members of `CoreGPUWorker`, constructed in the constructor body (after member data fields are initialized). They store raw pointers/references back to the worker's member data. Worker outlives both components.

## 3. ANN Library Decomposition

### 3.1 New: `ANN_GPUBufferManager.hpp` / `ANN_GPUBufferManager.cpp`

**~280 lines .cpp**

```cpp
namespace ANN {
  template <typename T>
  class GPUBufferManager {
  public:
    GPUBufferManager(OpenCLWrapper::Core* core, const LayersConfig& layersConfig,
                     Parameters<T>& parameters, const TrainingConfig<T>& trainingConfig,
                     LogLevel logLevel);

    // Initialization
    void initializeParameters();
    void loadSources(bool skipDefines);
    void allocateBuffers();

    // Parameter synchronization
    void syncParametersFromGPU();

    // Data I/O
    Output<T> readOutput();
    Tensor1D<T> readInputGradients();

    // Gradient access (for multi-GPU merging)
    void readAccumulatedGradients(Tensor1D<T>& accumWeights, Tensor1D<T>& accumBiases);
    void setAccumulators(const Tensor1D<T>& accumWeights, const Tensor1D<T>& accumBiases);

    // Dropout
    void generateAndUploadDropoutMask();

    // Offset queries (used by CNN integration)
    ulong getOutputActvOffset() const;
    ulong getNumOutputNeurons() const;

  private:
    OpenCLWrapper::Core* core;
    const LayersConfig& layersConfig;
    Parameters<T>& parameters;
    const TrainingConfig<T>& trainingConfig;
    LogLevel logLevel;
    std::mt19937 dropoutRng{std::random_device{}()};

    ulong getActvOffset(ulong layerIdx) const;
    ulong getWeightOffset(ulong layerIdx) const;
    ulong getBiasOffset(ulong layerIdx) const;
  };
}
```

**Methods migrated from `CoreGPUWorker`:**
- `initializeParameters()` (lines 299–354)
- `loadSources()` (lines 358–381)
- `allocateBuffers()` (lines 389–463)
- `syncParametersFromGPU()` (lines 278–296)
- `readOutput()` (lines 922–942)
- `readInputGradients()` (lines 904–918)
- `readAccumulatedGradients()` (lines 228–240)
- `setAccumulators()` (lines 246–251)
- `generateAndUploadDropoutMask()` (lines 948–979)
- `getActvOffset()`, `getWeightOffset()`, `getBiasOffset()` (lines 857–881)
- `getOutputActvOffset()`, `getNumOutputNeurons()` (lines 883–893)



### 3.2 New: `ANN_GPUKernelBuilder.hpp` / `ANN_GPUKernelBuilder.cpp`

**~310 lines .cpp**

```cpp
namespace ANN {
  template <typename T>
  class GPUKernelBuilder {
  public:
    GPUKernelBuilder(OpenCLWrapper::Core* core, const LayersConfig& layersConfig,
                     const Parameters<T>& parameters, const TrainingConfig<T>& trainingConfig,
                     const CostFunctionConfig<T>& costFunctionConfig, bool hasDropout);

    // Kernel building blocks (public — CNN calls these via ANN::CoreGPUWorker delegation)
    void addPropagateKernels();
    void addBackpropagateKernels(bool includeInputGradients);
    void addAccumulateKernels();
    void addUpdateKernels(ulong numSamples);

    // Kernel setup orchestration
    void setupPredictKernels();
    void setupTrainingKernels();
    void setupBackpropagateKernels();
    void setupAccumulateKernels();
    void setupUpdateKernels(ulong numSamples);

    void invalidateAllKernelFlags();

    // Kernel flags (public so worker can check them)
    bool predictKernelsSetup = false;
    bool trainingKernelsSetup = false;
    bool backpropagateKernelsSetup = false;
    bool accumulateKernelsSetup = false;
    bool updateKernelsSetup = false;

  private:
    OpenCLWrapper::Core* core;
    const LayersConfig& layersConfig;
    const Parameters<T>& parameters;
    const TrainingConfig<T>& trainingConfig;
    CostFunctionConfig<T> costFunctionConfig;
    bool hasDropout;
    ulong adam_t = 0;

    ulong getActvOffset(ulong layerIdx) const;
    ulong getWeightOffset(ulong layerIdx) const;
    ulong getBiasOffset(ulong layerIdx) const;
  };
}
```

**Methods migrated from `CoreGPUWorker`:**
- `addPropagateKernels()` (lines 601–663)
- `addBackpropagateKernels()` (lines 667–817)
- `addAccumulateKernels()` (lines 821–836)
- `addUpdateKernels()` (lines 538–595)
- `setupPredictKernels()` (lines 469–478)
- `setupTrainingKernels()` (lines 482–493)
- `setupBackpropagateKernels()` (lines 497–507)
- `setupAccumulateKernels()` (lines 511–520)
- `setupUpdateKernels()` (lines 524–534)
- `invalidateAllKernelFlags()` (lines 843–850)
- Offset helpers (lines 857–881) — duplicated, 3 trivial functions


### 3.3 Modified: `ANN_CoreGPUWorker.hpp` / `ANN_CoreGPUWorker.cpp`

**~200 lines .cpp (down from 985)**

The worker becomes a thin coordinator:
1. Owns the `OpenCLWrapper::Core` (owned or shared pointer pattern — unchanged)
2. Owns `GPUBufferManager<T>` and `GPUKernelBuilder<T>` as `std::unique_ptr` members
3. Contains all execution logic (predict, train, test, step-by-step, update)
4. Provides delegation wrappers for the public API used by CNN integration

**Public API remains identical** — all existing callers continue to work. Delegation wrappers
are inline one-liners in the header:

```cpp
// Example delegation wrappers (in header)
void loadSources(bool skip)         { bufferManager->loadSources(skip); }
void addPropagateKernels()          { kernelBuilder->addPropagateKernels(); }
Output<T> readOutput()              { return bufferManager->readOutput(); }
ulong getOutputActvOffset() const   { return bufferManager->getOutputActvOffset(); }
```

**What stays in `CoreGPUWorker.cpp`:**
- Constructors (lines 15–54) — create core, create components, call init sequence
- `predict()` (lines 60–76)
- `trainSubset()` (lines 80–133)
- `testSubset()` (lines 137–174)
- `backpropagate()` (lines 179–195)
- `accumulate()` (lines 199–209)
- `resetAccumulators()` (lines 213–225)
- `update()` (lines 257–272)

**Constructor change example:**
```cpp
CoreGPUWorker<T>::CoreGPUWorker(...) : layersConfig(...), ... {
  this->ownedCore = std::make_unique<OpenCLWrapper::Core>(false);
  this->core = this->ownedCore.get();
  this->core->setVerbose(this->logLevel >= LogLevel::DEBUG);

  this->bufferManager = std::make_unique<GPUBufferManager<T>>(
      this->core, this->layersConfig, this->parameters,
      this->trainingConfig, this->logLevel);
  this->kernelBuilder = std::make_unique<GPUKernelBuilder<T>>(
      this->core, this->layersConfig, this->parameters, this->trainingConfig,
      this->costFunctionConfig, this->hasDropout);

  this->bufferManager->initializeParameters();
  this->bufferManager->loadSources(false);
  this->bufferManager->allocateBuffers();
}
```

**Execution methods reference components instead of `this`:**
```cpp
Output<T> CoreGPUWorker<T>::predict(const Input<T>& input) {
  if (!this->kernelBuilder->predictKernelsSetup) {
    this->kernelBuilder->setupPredictKernels();
  }
  this->core->template writeBuffer<T>("actvs", input, 0);
  this->core->run();
  return this->bufferManager->readOutput();
}
```


## 4. CNN Library Decomposition

### 4.1 New: `CNN_GPUBufferManager.hpp` / `CNN_GPUBufferManager.cpp`

**~310 lines .cpp**

```cpp
namespace CNN {
  template <typename T>
  class GPUBufferManager {
  public:
    // Nested types (needed by kernel builder)
    struct LayerInfo { ulong actvOffset; ulong actvSize; };
    struct ConvInfo  { ulong filterOffset; ulong biasOffset; ulong numFilterElems; ulong numBiases; };
    struct PoolInfo  { ulong indexOffset; ulong indexSize; };

    GPUBufferManager(OpenCLWrapper::Core* core, const CoreConfig<T>& coreConfig,
                     Parameters<T>& parameters, LogLevel logLevel);

    // Initialization
    void computeLayerOffsets();
    void loadSources(bool skipDefines);
    void allocateBuffers();
    void buildANNWorker();

    // Parameter synchronization
    void syncParametersFromGPU();

    // Gradient access (CNN)
    void readAccumulatedGradients(std::vector<T>& accumFilters, std::vector<T>& accumBiases);
    void setAccumulators(const std::vector<T>& accumFilters, const std::vector<T>& accumBiases);

    // Gradient access (ANN delegation)
    void readANNAccumulatedGradients(ANN::Tensor1D<T>& accumW, ANN::Tensor1D<T>& accumB);
    void setANNAccumulators(const ANN::Tensor1D<T>& accumW, const ANN::Tensor1D<T>& accumB);

    // Layout data accessors (used by kernel builder)
    const std::vector<LayerInfo>& getLayerInfos() const { return layerInfos; }
    const std::vector<ConvInfo>& getConvInfos() const   { return convInfos; }
    const std::vector<PoolInfo>& getPoolInfos() const   { return poolInfos; }
    ulong getTotalActvSize() const      { return totalActvSize; }
    ulong getTotalFilterSize() const    { return totalFilterSize; }
    ulong getTotalBiasSize() const      { return totalBiasSize; }
    ulong getTotalPoolIndexSize() const { return totalPoolIndexSize; }
    ulong getFlattenSize() const        { return flattenSize; }

    // ANN worker access (used by kernel builder for pipeline setup)
    ANN::CoreGPUWorker<T>* getANNWorker() { return annGPUWorker.get(); }

  private:
    OpenCLWrapper::Core* core;
    const CoreConfig<T>& coreConfig;
    Parameters<T>& parameters;
    LogLevel logLevel;

    Shape3D cnnOutputShape;
    ulong flattenSize;

    std::vector<LayerInfo> layerInfos;
    ulong totalActvSize = 0;
    std::vector<ConvInfo> convInfos;
    ulong totalFilterSize = 0;
    ulong totalBiasSize = 0;
    std::vector<PoolInfo> poolInfos;
    ulong totalPoolIndexSize = 0;

    std::unique_ptr<ANN::CoreGPUWorker<T>> annGPUWorker;
  };
}
```

**Methods migrated from `CoreGPUWorker`:**
- `computeLayerOffsets()` (lines 84–164)
- `loadSources()` (lines 171–190)
- `allocateBuffers()` (lines 200–280)
- `buildANNWorker()` (lines 287–333)
- `syncParametersFromGPU()` (lines 1090–1122)
- `readAccumulatedGradients()` (lines 1002–1016)
- `setAccumulators()` (lines 1018–1030)
- `readANNAccumulatedGradients()` (lines 1035–1039)
- `setANNAccumulators()` (lines 1044–1048)


### 4.2 New: `CNN_GPUKernelBuilder.hpp` / `CNN_GPUKernelBuilder.cpp`

**~475 lines .cpp**

```cpp
namespace CNN {
  template <typename T>
  class GPUKernelBuilder {
  public:
    GPUKernelBuilder(OpenCLWrapper::Core* core, const CoreConfig<T>& coreConfig,
                     GPUBufferManager<T>& bufferManager);

    // Kernel building blocks (public for shared-core integration)
    void addPropagateKernels();
    void addBackpropagateKernels();
    void addCNNAccumulateKernels();
    void addCNNUpdateKernels(ulong numSamples);
    void addCopyBridgeKernels();

    // Kernel setup orchestration
    void setupPredictKernels();
    void setupTrainingKernels();
    void setupUpdateKernels(ulong numSamples);

    void invalidateAllKernelFlags();

    // Kernel flags
    bool predictKernelsSetup = false;
    bool trainingKernelsSetup = false;
    bool updateKernelsSetup = false;

  private:
    OpenCLWrapper::Core* core;
    const CoreConfig<T>& coreConfig;
    GPUBufferManager<T>& bufferManager;  // for layout data + ANN worker access
    ulong adam_t = 0;
  };
}
```

**Key design note:** The kernel builder holds a reference to `GPUBufferManager` to access:
- `getLayerInfos()`, `getConvInfos()`, `getPoolInfos()` — for kernel argument setup
- `getTotalFilterSize()`, `getTotalBiasSize()` — for accumulate/update kernels
- `getFlattenSize()` — for bridge kernels
- `getANNWorker()` — for `setupTrainingKernels()` which calls ANN's `addPropagateKernels()`, etc.

This is a one-way dependency (kernel builder → buffer manager), not circular.

**Methods migrated from `CoreGPUWorker`:**
- `addPropagateKernels()` (lines 339–435)
- `addBackpropagateKernels()` (lines 441–613)
- `addCNNAccumulateKernels()` (lines 619–637)
- `addCNNUpdateKernels()` (lines 643–712)
- `addCopyBridgeKernels()` (lines 718–729)
- `setupPredictKernels()` (lines 733–747)
- `setupTrainingKernels()` (lines 753–796)
- `setupUpdateKernels()` (lines 798–812)
- `invalidateAllKernelFlags()` (lines 814–824)

### 4.3 Modified: `CNN_CoreGPUWorker.hpp` / `CNN_CoreGPUWorker.cpp`

**~250 lines .cpp (down from 1137)**

Same facade pattern as ANN. The worker:
1. Owns `OpenCLWrapper::Core` (owned or shared)
2. Owns `GPUBufferManager<T>` and `GPUKernelBuilder<T>` as `std::unique_ptr` members
3. Contains execution logic + kernel save/restore
4. Provides delegation wrappers for public API

**What stays in `CoreGPUWorker.cpp`:**
- Constructors (lines 23–78)
- `predict()` (lines 830–848)
- `trainSubset()` (lines 855–912)
- `testSubset()` (lines 914–938)
- `backpropagateSample()` (lines 943–997)
- `accumulate()` (lines 972–985) — calls `core->run()` with accumulate kernels
- `resetAccumulators()` (lines 987–997)
- `update()` (lines 1054–1060)
- `saveKernels()`, `restoreKernels()`, `setTrainingKernelsReady()` (lines 1066–1084)

**Delegation wrappers in header:**
```cpp
void loadSources(bool skip)         { bufferManager->loadSources(skip); }
void allocateBuffers()              { bufferManager->allocateBuffers(); }
void addPropagateKernels()          { kernelBuilder->addPropagateKernels(); }
void addCopyBridgeKernels()         { kernelBuilder->addCopyBridgeKernels(); }
void addBackpropagateKernels()      { kernelBuilder->addBackpropagateKernels(); }
void addCNNAccumulateKernels()      { kernelBuilder->addCNNAccumulateKernels(); }
void addCNNUpdateKernels(ulong n)   { kernelBuilder->addCNNUpdateKernels(n); }
void syncParametersFromGPU()        { bufferManager->syncParametersFromGPU(); }
// etc.
```

**Constructor change:**
```cpp
CoreGPUWorker<T>::CoreGPUWorker(const CoreConfig<T>& config) : coreConfig(config), ... {
  this->ownedCore = std::make_unique<OpenCLWrapper::Core>(false);
  this->core = this->ownedCore.get();

  this->bufferManager = std::make_unique<GPUBufferManager<T>>(
      this->core, this->coreConfig, this->parameters, this->logLevel);
  this->bufferManager->computeLayerOffsets();
  this->bufferManager->loadSources(false);
  this->bufferManager->buildANNWorker();
  this->bufferManager->allocateBuffers();

  this->kernelBuilder = std::make_unique<GPUKernelBuilder<T>>(
      this->core, this->coreConfig, *this->bufferManager);
}
```


## 5. File Changes Summary

### New Files (8 total)
| File | Est. Lines (.cpp) | Responsibility |
|------|-------------------|----------------|
| `extern/CNN/extern/ANN/ANN_GPUBufferManager.hpp` | ~60 | ANN buffer manager header |
| `extern/CNN/extern/ANN/ANN_GPUBufferManager.cpp` | ~280 | Buffer alloc, source loading, param init/sync, data I/O, offsets, dropout |
| `extern/CNN/extern/ANN/ANN_GPUKernelBuilder.hpp` | ~55 | ANN kernel builder header |
| `extern/CNN/extern/ANN/ANN_GPUKernelBuilder.cpp` | ~310 | Kernel building blocks, setup orchestration |
| `extern/CNN/CNN_GPUBufferManager.hpp` | ~65 | CNN buffer manager header |
| `extern/CNN/CNN_GPUBufferManager.cpp` | ~310 | Offset computation, buffer alloc, ANN worker, param sync, gradient access |
| `extern/CNN/CNN_GPUKernelBuilder.hpp` | ~40 | CNN kernel builder header |
| `extern/CNN/CNN_GPUKernelBuilder.cpp` | ~475 | CNN kernel building blocks, setup orchestration |

### Modified Files (4 total)
| File | Before | After | Change |
|------|--------|-------|--------|
| `extern/CNN/extern/ANN/ANN_CoreGPUWorker.hpp` | 130 lines | ~70 lines | Replace method declarations with delegation wrappers; add component members |
| `extern/CNN/extern/ANN/ANN_CoreGPUWorker.cpp` | 985 lines | ~200 lines | Keep only constructors + execution methods |
| `extern/CNN/CNN_CoreGPUWorker.hpp` | 151 lines | ~80 lines | Replace method declarations with delegation wrappers; add component members |
| `extern/CNN/CNN_CoreGPUWorker.cpp` | 1137 lines | ~250 lines | Keep only constructors + execution + save/restore |

### Build System
- Add new `.cpp` files to the build system (CMakeLists.txt or .pro file)
- No new external dependencies

## 6. Implementation Steps

### Step 1: ANN GPUBufferManager (lowest risk, no dependencies on kernel builder)
1. Create `ANN_GPUBufferManager.hpp` with class declaration
2. Create `ANN_GPUBufferManager.cpp`, move methods from `ANN_CoreGPUWorker.cpp`
3. Change method prefixes from `CoreGPUWorker<T>::` to `GPUBufferManager<T>::`
4. Change `this->` references for moved member variables
5. Add `#include "ANN_GPUBufferManager.hpp"` to `ANN_CoreGPUWorker.hpp`
6. Add `std::unique_ptr<GPUBufferManager<T>> bufferManager` member to worker
7. Update constructors to create buffer manager, delegate init calls
8. Replace direct calls with `bufferManager->` calls in execution methods
9. Build and verify

### Step 2: ANN GPUKernelBuilder
1. Create `ANN_GPUKernelBuilder.hpp` with class declaration
2. Create `ANN_GPUKernelBuilder.cpp`, move kernel building + setup methods
3. Duplicate the 3 trivial offset helper functions (they're pure functions of layersConfig)
4. Add `std::unique_ptr<GPUKernelBuilder<T>> kernelBuilder` member to worker
5. Update constructors to create kernel builder
6. Replace kernel calls with `kernelBuilder->` calls
7. Update public API methods to delegate to kernel builder
8. Build and verify ANN library compiles and links

### Step 3: CNN GPUBufferManager
1. Create `CNN_GPUBufferManager.hpp` — move `LayerInfo`, `ConvInfo`, `PoolInfo` structs here
2. Create `CNN_GPUBufferManager.cpp`, move init + IO methods
3. Move `annGPUWorker` ownership from worker to buffer manager
4. Add const accessor methods for layout data
5. Update worker to create buffer manager, delegate calls
6. Build and verify

### Step 4: CNN GPUKernelBuilder
1. Create `CNN_GPUKernelBuilder.hpp`
2. Create `CNN_GPUKernelBuilder.cpp`, move kernel building + setup methods
3. Replace `this->layerInfos` etc. with `this->bufferManager.getLayerInfos()` etc.
4. Replace `this->annGPUWorker->` with `this->bufferManager.getANNWorker()->`
5. Update worker to create kernel builder, delegate calls
6. Build and verify full CNN+ANN pipeline works

### Step 5: Final cleanup
1. Remove dead code from slimmed worker files
2. Verify all template instantiations are in the new `.cpp` files
3. Run full test suite
4. Clean up includes (remove unused headers from worker files)

## 7. Testing Strategy

- **Build verification** after each step (incremental — each step should compile)
- **Existing tests** should pass unchanged since the public API is identical
- **No new tests needed** for this refactoring (behavior-preserving)
- **Manual test:** Run a predict + train cycle for both ANN and CNN to verify GPU execution works

## 8. Key Design Decisions & Rationale

### Why composition over inheritance?
- The responsibilities (buffer management, kernel building) are not "is-a" relationships
- Inheritance would create a diamond problem if CNN inherits from both buffer+kernel mixins
- Composition gives clearest ownership semantics

### Why `std::unique_ptr` for components instead of direct members?
- Components store references back to the worker's member data (layersConfig, parameters, etc.)
- Direct members would be initialized in the member initializer list, before the data they reference
- `std::unique_ptr` allows deferred construction in the constructor body, after all data members

### Why duplicate offset helpers in ANN kernel builder?
- They're 3 functions of ~5 lines each, pure functions of `layersConfig`
- Avoids kernel builder depending on buffer manager (keeps ANN's two components independent)
- CNN kernel builder uses buffer manager's accessors instead (it already depends on buffer manager)

### Why CNN kernel builder depends on buffer manager?
- CNN's kernel setup orchestration (`setupTrainingKernels`) needs to call
  `annGPUWorker->addPropagateKernels()`, which is owned by the buffer manager
- Also needs layout data (layerInfos, convInfos, etc.) from buffer manager
- One-way dependency is acceptable; avoids passing 8+ parameters to every method

## 9. Rollback Plan

- Each step produces a compilable state — can stop at any step
- All changes are internal (no public API changes) — revert is simply restoring files
- Git branch recommended: `refactor/coregpuworker-decomposition`

## 10. Estimated Effort

- **Steps 1-2 (ANN):** ~3-4 hours (simpler structure, fewer cross-dependencies)
- **Steps 3-4 (CNN):** ~4-5 hours (more complex due to ANN worker integration)
- **Step 5 (cleanup):** ~1 hour
- **Total:** ~8-10 hours
- **Complexity:** Medium — purely mechanical refactoring with clear boundaries
