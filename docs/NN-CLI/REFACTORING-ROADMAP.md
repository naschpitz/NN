# NN-CLI Refactoring Roadmap

**Branch:** `refactor/analysis-deep-dive`
**Base commit:** `d5a888f`
**Date:** 2026-06-07
**Analysis scope:** 5 projects (NN-CLI, CNN, ANN, OpenCLWrapper, NN-Server context), 164 types inventoried (64 class, 80 struct, 20 enum class), ~25,000+ lines analyzed

---

## Executive Summary

This document presents a comprehensive, prioritized refactoring roadmap for the NN-CLI project and its submodule dependencies (CNN, ANN, OpenCLWrapper). The analysis combined three independent audits:

1. **Architectural analysis** — class responsibilities, coupling patterns, boundary correctness
2. **Code review audit** — best practices, security, correctness, conformance to AGENTS.md standards
3. **DRY/duplication analysis** — systematic identification of copy-paste and structural duplication

**Overall code quality score: C+ (70/100)**

The codebase scores well on individual file organization, RAII usage, template design, and adherence to the "no caller-side type branching" rule (0 violations). However, it is significantly weighed down by massive structural duplication between the ANN and CNN libraries and their corresponding NN-CLI Runner classes.

**Total identified duplication: ~1,500+ lines** of avoidable duplicate code, with additional structural similarity across ~600 more lines.

---

## Priorities at a Glance

| Phase | Focus | Items | Effort | Impact |
|-------|-------|-------|--------|--------|
| **1** | Immediate Bug Fixes | 2 | Hours | CRITICAL - fixes bugs and security vulns |
| **2** | Quick Wins | 7 | Days | HIGH - improves code quality immediately |
| **3** | Structural DRY | 12 | Weeks | MEDIUM - eliminates largest duplication sources |
| **4** | Long-term Architecture | 2 | Months | MEDIUM - foundational architectural improvements |

---

## Phase 1: Immediate Bug Fixes

These are correctness bugs and security vulnerabilities that should be fixed before any other work.

### 1.1 Static Local Variables in Training Callbacks — CRITICAL

**Files:** `NN-CLI_ANNRunner.cpp:482-491`, `NN-CLI_CNNRunner.cpp:560-569`

**Problem:** Four `static` local variables are declared inside `setupTrainingCallback()` in both runners:
```cpp
static ulong lastCallbackEpoch = 0;
static float lastEpochLoss = 0.0f;
static std::mutex epochTransitionMutex;
static ProgressBar progressBar(this->ioConfig.progressReports, 50, std::max(2UL, batchSize / 2));
```

These are **shared across ALL instances** of the respective Runner class. If two runners are created (concurrently or sequentially), the second instance silently corrupts the first's state. The static `std::mutex` creates potential for deadlock. The static `ProgressBar` is initialized once with the first call's `batchSize` and never updated (the code even has a comment acknowledging this).

**Fix:** Move all four variables to be instance members of `ANNRunner`/`CNNRunner`:
- `ulong lastCallbackEpoch_` initialized to `0`
- `float lastEpochLoss_` initialized to `0.0f`
- `std::mutex epochTransitionMutex_`
- `ProgressBar progressBar_` (or `std::optional<ProgressBar>`) initialized in `train()`

**Effort:** 1-2 hours per runner (low risk, no API change)

### 1.2 Command Injection via `std::system()` — HIGH

**Files:** `NN-CLI_CalibrateRunner.cpp:97-104` (shellRun function), with injection points at lines 286, 289, 305, 307

**Problem:** User-controlled `--ood-dir` CLI argument flows unsanitized into shell commands:
```cpp
void shellRun(const std::string& cmd, const std::string& description) {
    int rc = std::system(cmd.c_str());  // line 99
}
```
Called with single-quoted paths:
```cpp
shellRun("tar -xzf '" + tgz.string() + "' -C '" + destDir + "' ...", ...);  // line 289
```
A path containing a single quote character breaks out of the quoting.

**Fix:** Replace `std::system()` with `QProcess` using argument vectors (no shell interpretation):
```cpp
QProcess process;
process.start("tar", {"-xzf", tgz.string(), "-C", destDir, ...});
if (!process.waitForFinished(600000))
    throw std::runtime_error("... timed out");
```

**Effort:** 2-3 hours (moderate - involves changing 4 call sites)

---

## Phase 2: Quick Wins

These are isolated, low-risk improvements that can be implemented independently.

### 2.1 Consolidate `using ulong` to Shared Header — MEDIUM

**Files:** 11 headers each define `using ulong = unsigned long;` independently

**Files involved:**
- `NN-CLI_DataLoader.hpp:25`
- `NN-CLI_GpuAugmenter.hpp:17`
- `NN-CLI_TrainingSummary.hpp:16`
- `NN-CLI_LossReferenceTable.hpp:14`
- `NN-CLI_SummaryTable.hpp:10`
- `NN-CLI_ModelSerializer.hpp:17`
- `NN-CLI_AugmentationConfig.hpp:12`
- `NN-CLI_TestSummary.hpp:10`
- `NN-CLI_ValidationDatasetConfig.hpp:9`
- `NN-CLI_DataSplitter.hpp:11`
- `NN-CLI_PredictSummary.hpp:12`

Additionally, `NN-CLI_ProgressBar.hpp` USES `ulong` at 8+ lines but never defines it.

**Fix:** Create `NN-CLI_Types.hpp`:
```cpp
#ifndef NN_CLI_TYPES_HPP
#define NN_CLI_TYPES_HPP
namespace NN_CLI { using ulong = unsigned long; }
#endif
```
Then remove the duplicate definitions from 11 headers, replacing with `#include "NN-CLI_Types.hpp"`.

**Effort:** 1 hour

### 2.2 Fix `unique_ptr::release()` Workaround — MEDIUM

**Files:** `NN-CLI_CNNRunner.cpp:214`, `NN-CLI_ANNRunner.cpp:187`

**Problem:**
```cpp
validationCore = std::shared_ptr<CNN::Core<float>>(
    CNN::Core<float>::makeCore(validationCoreConfig).release());
```

If `makeCore` throws between allocation and shared_ptr construction, the raw pointer is leaked. Root cause: Runner constructors accept `std::unique_ptr<Core<T>>&` (non-const ref), forcing workarounds.

**Fix:** Change Runner constructors to accept `Core<T>&` or `Core<T>*` (the parent Runner already owns the Core). Then create shared_ptr from non-owning pointer:
```cpp
validationCore = std::shared_ptr<CNN::Core<float>>(
    CNN::Core<float>::makeCore(validationCoreConfig));  // implicit unique_ptr→shared_ptr conversion
```

**Effort:** 2 hours (affects Runner constructor signatures and callers)

### 2.3 Fix `const_cast` in `const` Method — LOW

**File:** `NN-CLI_TrainingProfiler.cpp:295`, `NN-CLI_TrainingProfiler.hpp:93`

**Problem:**
```cpp
const_cast<TrainingProfiler*>(this)->lastRenderedBatchNumber = v.batchNumber;  // .cpp:295
// ...
ulong lastRenderedBatchNumber = static_cast<ulong>(-1);  // .hpp:93 (not mutable)
```

**Fix:** Declare `lastRenderedBatchNumber` as `mutable` in the header:
```cpp
mutable ulong lastRenderedBatchNumber = static_cast<ulong>(-1);
```
Then remove the `const_cast` from the .cpp.

**Effort:** 15 minutes

### 2.4 Add Default Case to Serializer Switch — LOW

**File:** `NN-CLI_ModelSerializer.cpp:292-350`

**Problem:** Switch on `CNN::LayerType` has no `default` case. When new layer types are added, the serializer silently produces invalid JSON (no `"type"` field).

**Fix:** Add:
```cpp
default:
    throw std::runtime_error("Unknown CNN layer type in serializer: " +
                             std::to_string(static_cast<int>(layer.type)));
```

**Effort:** 15 minutes

### 2.5 Validate CLI Numeric Arguments — MEDIUM

**File:** `NN-CLI_CalibrateRunner.cpp:153-157`

**Problem:**
```cpp
std::size_t idCount =
    this->parser.isSet("id-sample-count") ? this->parser.value("id-sample-count").toULongLong() : 500;
```
`toULongLong()` and `toDouble()` are called without `bool* ok`. Invalid input like `--id-sample-count abc` silently becomes 0.

**Fix:**
```cpp
bool ok = false;
std::size_t idCount = 500;
if (this->parser.isSet("id-sample-count")) {
    idCount = this->parser.value("id-sample-count").toULongLong(&ok);
    if (!ok) throw std::runtime_error("Error: --id-sample-count must be a positive integer");
}
```

**Effort:** 1 hour

### 2.6 Extract Shared CMake Sources — MEDIUM

**File:** `CMakeLists.txt` (lines 37-62 and 128-158)

**Problem:** 10 `.cpp` files are compiled twice (once for NN-CLI, once for test_nncli): `ANNLoader`, `DataLoader`, `GpuAugmenter`, `DataSplitter`, `DataType`, `ImageLoader`, `Loader`, `ProgressBar`, `SummaryTable`, `TrainingSummary`.

**Fix:** Extract shared sources into a CMake OBJECT library:
```cmake
add_library(nncli_common OBJECT
    NN-CLI_ANNLoader.cpp NN-CLI_DataLoader.cpp NN-CLI_GpuAugmenter.cpp
    NN-CLI_DataSplitter.cpp NN-CLI_DataType.cpp NN-CLI_ImageLoader.cpp
    NN-CLI_Loader.cpp NN-CLI_ProgressBar.cpp NN-CLI_SummaryTable.cpp
    NN-CLI_TrainingSummary.cpp
)
target_link_libraries(NN-CLI PRIVATE nncli_common)
target_link_libraries(test_nncli PRIVATE nncli_common)
```

**Effort:** 1 hour

### 2.7 Replace C-Style Casts with static_cast — LOW

**File:** `NN-CLI_GpuAugmenter.cpp` (lines 143, 144, 175, 176, 177, 178, 179, 180, 181, 205, 210, 418, 448) — ~13 C-style casts.

**Fix:** Replace all `(int)`, `(long)`, `(float)` C-style casts with `static_cast<>`.

**Effort:** 1 hour

---

## Phase 3: Structural DRY Refactoring

These are larger efforts that eliminate the most significant duplication.

### 3.1 Extract Shared Config Structs — MEDIUM EFFORT, HIGH IMPACT

**Files involved (18 ANN/CNN file-pairs, ~720 lines total):**
- `ANN/ANN_MonitoringConfig.hpp` ↔ `CNN/CNN_MonitoringConfig.hpp` (35 lines each)
- `ANN/ANN_TrainingConfig.hpp` ↔ `CNN/CNN_TrainingConfig.hpp` (34 lines each)
- `ANN/ANN_CostFunctionConfig.hpp` ↔ `CNN/CNN_CostFunctionConfig.hpp` (53 lines each)
- `ANN/ANN_TestConfig.hpp` ↔ `CNN/CNN_TestConfig.hpp` (17 lines each)
- `ANN/ANN_Optimizer.hpp` ↔ `CNN/CNN_Optimizer.hpp` (30 lines each)
- `ANN/ANN_Device.hpp` ↔ `CNN/CNN_Device.hpp` (28 lines each)
- `ANN/ANN_Mode.hpp` ↔ `CNN/CNN_Mode.hpp` (29 lines each)
- `ANN/ANN_ValidationConfig.hpp` ↔ `CNN/CNN_ValidationConfig.hpp` (20 lines each)
- `ANN/ANN_LogLevel.hpp` ↔ `CNN/CNN_LogLevel.hpp` (13 lines each)
- `ANN/ANN_TestResult.hpp` ↔ `CNN/CNN_TestResult.hpp` (22 lines each)
- `ANN/ANN_PredictMetadata.hpp` ↔ `CNN/CNN_PredictMetadata.hpp` (21-22 lines each)
- `ANN/ANN_PredictResult.hpp` ↔ `CNN/CNN_PredictResult.hpp` (31 lines each)
- `ANN/ANN_TrainingProgress.hpp` ↔ `CNN/CNN_TrainingProgress.hpp` (33-37 lines each)
- `ANN/ANN_ProgressCallback.hpp` ↔ `CNN/CNN_ProgressCallback.hpp` (18 lines each)
- `ANN/ANN_TrainingMetadata.hpp` ↔ `CNN/CNN_TrainingMetadata.hpp` (30 lines each)
- `ANN/ANN_TrainingMonitor.hpp` ↔ `CNN/CNN_TrainingMonitor.hpp` (48 lines each)
- `ANN/ANN_Optimizer.cpp` ↔ `CNN/CNN_Optimizer.cpp` (43 lines each)
- `ANN/ANN_TrainingMonitor.cpp` ↔ `CNN/CNN_TrainingMonitor.cpp` (153 lines each)

**Strategy:** These are 100% identical except for namespace prefix and include guard. Create a shared location (e.g., `NN-Common` library or have one library own them and the other re-export). Since AGENTS.md dictates that changes go in original repos and submodules are updated:
1. In the ANN repo, create `ANN/Common/` directory with the shared configs
2. In the CNN repo, replace CNN's copies with `#include` of ANN's versions + `using` declarations
3. Update submodules in NN-CLI

**Effort:** 2-3 days (due to submodule workflow, not code complexity)

### 3.2 Unify TrainingMonitor — LOW EFFORT, HIGH IMPACT

**Files:** `ANN_TrainingMonitor.cpp` (153 lines) ↔ `CNN_TrainingMonitor.cpp` (153 lines) — 100% identical

**Fix:** Have CNN's TrainingMonitor simply include ANN's version:
```cpp
// CNN_TrainingMonitor.hpp
#include <ANN_TrainingMonitor.hpp>
namespace CNN { template<typename T> using TrainingMonitor = ANN::TrainingMonitor<T>; }
```
Then remove `CNN_TrainingMonitor.cpp` from build.

**Effort:** 1 hour

### 3.3 Extract CalibrateRunner::runPredict() ANN/CNN Fork — MEDIUM

**File:** `NN-CLI_CalibrateRunner.cpp:494-610`

**Problem:** Two ~58-line branches that are structurally identical except for type names (`CNN::Inputs<float>` vs `ANN::Inputs<float>`, `CNN::Input<float>` wrapper vs raw vector).

**Fix:** Extract shared loading/progress logic into a helper function that accepts the type-specific parts via template or lambda. Example approach:
```cpp
template<typename InputT>
std::vector<InputT> loadAndPredict(
    Core<float>& core, const std::vector<ImageBatch>& batches,
    ProgressBar& progressBar, ulong totalSamples);
```

**Effort:** 4-6 hours

### 3.4 Extract ANNRunner ↔ CNNRunner Shared Methods — HIGH

**Files:** `NN-CLI_ANNRunner.cpp` (692 lines) ↔ `NN-CLI_CNNRunner.cpp` (790 lines)

**Problem:** ~500-600 lines of near-identical code across `train()`, `test()`, `predict()`, `setupTrainingCallback()`, `finishTraining()`, `computeClassWeightsFromOutputs()`, `loadSamplesFromOptions()`.

**Recommended sub-steps (in order):**

3.4.1 **Extract `computeClassWeightsFromOutputs()`** — Already 100% identical static method in both runners (27 lines). Move to `NN-CLI_Utils.hpp`. Effort: 30 min.

3.4.2 **Extract `ValidationState` struct** — 100% identical struct defined in both headers (8 lines). Move to shared header. Effort: 15 min.

3.4.3 **Extract `finishTraining()` helpers** — Structurally identical except for `saveANNModel`/`saveCNNModel` call. Extract path generation and model-saving dispatch. Effort: 2-3 hours.

3.4.4 **Extract `predict()` output logic** — ~110 shared lines of output-path resolution, image output, JSON serialization. Effort: 4-6 hours.

3.4.5 **Extract `loadSamplesFromOptions()`** — ~50 lines each, ~94% structurally identical. Effort: 2-3 hours.

3.4.6 **Consider `RunnerBase<CoreT, SampleT, ConfigT>` template** — After extracting all free functions, evaluate if remaining runner-specific code justifies a template base class. Effort: 1-2 days.

**Total effort: 3-5 days**

### 3.5 Extract ModelSerializer Shared Serialization — LOW

**File:** `NN-CLI_ModelSerializer.cpp` (554 lines)

**Problem:** `saveANNModel()` (lines 130-244) and `saveCNNModel()` (lines 250-489) share ~60 lines of identical serialization for training config, monitoring config, test config, and training metadata.

**Fix:** Extract shared serialization methods: `serializeTrainingConfig()`, `serializeMonitoringConfig()`, `serializeTestConfig()`, `serializeTrainingMetadata()`.

**Effort:** 3-4 hours

### 3.6 Extract Device String Helper — LOW

**Files:** `NN-CLI_TrainingSummary.cpp:131-139` and `315-321`, `NN-CLI_TestSummary.cpp:35-41` and `85-91`, `NN-CLI_PredictSummary.cpp:36-42` and `75-81`

**Problem:** Device string like "GPU (2x)" or "CPU (4 threads)" constructed identically in 4 places (8 occurrences total).

**Fix:** Extract to `SummaryTable::deviceString(DeviceType, int numDevices, int numThreads)`.

**Effort:** 1 hour

### 3.7 Extract Output Path Utilities — LOW

**Files:** `NN-CLI_ModelSerializer.cpp:504-552`, `NN-CLI_ANNRunner.cpp:285-299`, `NN-CLI_CNNRunner.cpp` (predict method)

**Problem:** The pattern "check if output dir exists, create if not" is repeated 5+ times.

**Fix:** Create `OutputDirResolver::ensureOutputDir(const QString& inputFilePath)`.

**Effort:** 1 hour

### 3.8 Extract CoreGPU test() Batch Distribution — MEDIUM

**Files:** `ANN_CoreGPU.cpp:306-371` ↔ `CNN_CoreGPU.cpp:335-400` — 100% identical

**Problem:** The `GPUWorkItem` struct, work distribution, `QtConcurrent::blockingMap` dispatch, and result aggregation are character-for-character identical.

**Fix:** Extract shared GPU test/predict batch distribution logic into a utility template in OpenCLWrapper or a shared location.

**Effort:** 2-3 hours

### 3.9 Fix Config File Re-parsing — MEDIUM

**File:** `NN-CLI_Runner.cpp:23-103`

**Problem:** Config file parsed 6 times independently.

**Fix:** Parse once in Runner constructor, pass `nlohmann::json` to all loaders:
```cpp
auto config = Loader::loadAllConfigs(configPath.toStdString());
this->networkType = config.networkType;
this->ioConfig = config.ioConfig;
this->augConfig = config.augConfig;
```

Or simpler: parse once, pass the json object:
```cpp
auto json = Loader::parseConfigFile(configPath);
this->networkType = Loader::detectNetworkType(json);
this->ioConfig = Loader::loadIOConfig(json);
```

**Effort:** 3-4 hours

### 3.10 Extract Miscellaneous Pattern Duplications — LOW

- Progress bar setup pattern (~32 lines, 4×): Wrap in `setupProgressCallback()` helper. Effort: 1-2 hours
- samples/idx-data validation guards (~18 lines, 6×): Centralize validation in `Runner` base. Effort: 1 hour
- Validation progress callback lambda (~8 lines, 2×): Extract lambda factory. Effort: 30 min

---

## Phase 4: Long-term Architecture

These are foundational changes that require careful planning and cross-repo coordination.

### 4.1 CoreBase Template Unification — HIGH EFFORT

**Files:** `ANN_Core.hpp`/`.cpp` ↔ `CNN_Core.hpp`/`.cpp`

**Problem:** ~240 lines of shared getter/setter/timing infrastructure duplicated between ANN and CNN Core classes. These share identical member names: `deviceType`, `modeType`, `numThreads`, `numGPUs`, `layersConfig`, `trainingConfig`, `testConfig`, `trainingMetadata`, `predictMetadata`, `parameters`, `progressReports`, `logLevel`, `trainingCallback`, `progressCallback`, `stopRequested`.

**Approach:** Create `CoreBase<T, CoreTraits>` template class with shared members and methods (getters, setters, timing). ANN::Core<T> and CNN::Core<T> inherit from it.

**Key challenge:** Both Cores have different interface methods (ANN has `backpropagate()`, `accumulate()`, `resetAccumulators()`, `update()`; CNN has `inputShape`, `TimingCallback`, `syncParametersToGPU()`). A traits pattern can handle these differences.

**Effort:** 1-2 weeks (requires coordinated changes to both ANN and CNN repos)

### 4.2 Runner Architecture Overhaul — HIGH EFFORT

**Problem:** The existing `Runner` class is a dispatcher that delegates to `ANNRunner`/`CNNRunner`. After Phase 3 extraction, remaining differences should be evaluated for whether a template-based or interface-based shared runner is justified.

**Considerations:**
- `ANN::Core<T>` vs `CNN::Core<T>` have different interfaces
- CNN has `inputShape`, `TrainingProfiler`, `GpuAugmenterPool` — ANN doesn't
- The free function extraction approach (Phase 3.4) may eliminate enough duplication to make templating unnecessary

**Recommendation:** Complete Phase 3 first, then re-evaluate. If 80%+ of runner code is shared, invest in template. If only 50%, accept the remaining duplication as the cost of supporting two different network types.

**Effort:** 1-2 weeks (after Phase 3)

---

## Architecture Dependency Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                          main.cpp                            │
└──────────────┬───────────────────────────────────────────────┘
               │ creates
               ▼
┌──────────────────────────────────────────────────────────────┐
│                       Runner (dispatcher)                     │
│  Owns: annCore, cnnCore, ioConfig, augConfig                 │
│  [ISSUE: Config parsed 6×]                                   │
│  Delegates to ANNRunner / CNNRunner / CalibrateRunner         │
└──────┬───────────────┬───────────────────┬───────────────────┘
       │               │                   │
       ▼               ▼                   ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐
│  ANNRunner   │ │  CNNRunner   │ │    CalibrateRunner        │
│  [BUG: static │ │ [BUG: static │ │ [VULN: std::system()]    │
│   locals]     │ │  locals]     │ │ [DRY: internal ANN/CNN   │
│ [DRY: ~500    │ │ [DRY: ~500   │ │  fork ~115 lines]        │
│  shared lines]│ │  shared lines]│ │                          │
└──────┬───────┘ └──────┬───────┘ └────────┬──────────────────┘
       │                │                  │
       │ Shared: TrainingTui, TerminalUI, ProgressBar,         │
       │ ModelSerializer, DataLoader<T>, ImageLoader           │
       ▼                ▼                  ▼
┌──────────────────────────────────────────────────────────────┐
│              NN-CLI Support Classes (47 types)                │
│  [ISSUE: ulong in 11 headers] [ISSUE: C-style casts]         │
│  [ISSUE: CMakeLists double-compiles 10 files]                │
└──────────────────────────┬───────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌─────────────────────────┐ ┌──────────────────────────┐
│     ANN Library         │ │      CNN Library           │
│  (39 types)             │ │   (74 types)               │
│                         │ │                            │
│  [DRY: 18 config file-  │ │ [DRY: same 18 file-pairs] │
│   pairs duplicated]     │ │                            │
│  TrainingMonitor 100%   │ │ TrainingMonitor 100% dup   │
│   identical to CNN      │ │                            │
│  Core getter/setter     │ │ Core getter/setter shared  │
│   ~240 lines shared]    │ │                            │
└────────┬────────────────┘ └────────┬───────────────────┘
         │                           │
         └──────────┬────────────────┘
                    ▼
         ┌────────────────────┐
         │  OpenCLWrapper     │
         │  (4 types)         │
         └────────────────────┘
```

---

## Summary of All Findings

| # | Finding | Severity | File(s) | Lines | Phase |
|---|---------|----------|---------|-------|-------|
| 1 | Static local vars in training callbacks | CRITICAL | ANNRunner.cpp, CNNRunner.cpp | 4 vars | 1 |
| 2 | Command injection via std::system() | HIGH | CalibrateRunner.cpp | 4 injection points | 1 |
| 3 | using ulong in 11 headers | MEDIUM | 11 NN-CLI headers | 11 defs | 2 |
| 4 | unique_ptr::release() leak risk | MEDIUM | CNNRunner.cpp, ANNRunner.cpp | 2 sites | 2 |
| 5 | const_cast in const method | MEDIUM | TrainingProfiler.cpp | 1 site | 2 |
| 6 | Missing default in serializer switch | LOW | ModelSerializer.cpp | 1 site | 2 |
| 7 | CLI arg validation | MEDIUM | CalibrateRunner.cpp | 3 sites | 2 |
| 8 | CMakeLists double compilation | MEDIUM | CMakeLists.txt | 10 files ×2 | 2 |
| 9 | C-style casts | LOW | GpuAugmenter.cpp | ~13 casts | 2 |
| 10 | Config struct duplication (18 pairs) | MEDIUM | ANN/CNN headers/cpp | ~720 lines | 3 |
| 11 | TrainingMonitor 100% duplicate | MEDIUM | ANN/CNN libs | 153 lines | 3 |
| 12 | CalibrateRunner internal ANN/CNN fork | MEDIUM | CalibrateRunner.cpp | ~115 lines | 3 |
| 13 | ANNRunner/CNNRunner shared code | HIGH | Both runners | ~500-600 lines | 3 |
| 14 | ModelSerializer save duplication | MEDIUM | ModelSerializer.cpp | ~60 lines | 3 |
| 15 | Device string generation (4×) | LOW | TrainingSummary, TestSummary, PredictSummary | ~28 lines | 3 |
| 16 | Output path resolution (5×) | LOW | ModelSerializer, both runners | ~30 lines | 3 |
| 17 | Config parsed 6× | MEDIUM | Runner.cpp | 6 calls | 3 |
| 18 | CoreGPU test() batch distribution | MEDIUM | ANN/CNN CoreGPU | ~66 lines | 3 |
| 19 | CoreBase getter/setter/timing | MEDIUM | ANN/CNN Core | ~240 lines | 4 |
| 20 | Runner architecture overhaul | MEDIUM | All runners | ~600+ lines | 4 |

---

## Appendix: Architectural Strengths

While this report focuses on issues, the codebase has genuine strengths worth preserving:

1. **No caller-side type branching violations** — The AGENTS.md rule is well-followed. Activation function and layer type dispatch happens internally via switch, never in callers.

2. **Good RAII usage** — Smart pointers, `QMutexLocker`, and `std::lock_guard` are used consistently.

3. **Clean header organization** — `public → protected → private` ordering is followed. Section headers in complex files.

4. **Template design with explicit instantiations** — Template classes follow the `.hpp` declaration → `.cpp` implementation → `template class X<T>` pattern correctly.

5. **Separation of concerns** — CNN layers (Conv2D, Pool, ReLU, etc.) are individually well-encapsulated.

6. **TrainingTui composition pattern** — Already a shared component between runners via composition rather than inheritance, showing good design judgment.
