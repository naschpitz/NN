#ifndef CNN_TIMINGCALLBACK_HPP
#define CNN_TIMINGCALLBACK_HPP

#include <functional>

//===================================================================================================================//
//-- Timing instrumentation callback --//
//
// The library does NOT measure or accumulate anything itself. It merely NOTIFIES
// the consumer (e.g. NN-CLI) when a measurable phase begins and ends, so that the
// consumer can timestamp the boundaries and build whatever profile it wants.
//
// This keeps all measurement/aggregation/display logic outside the library and the
// library change minimal: a handful of emitTiming(...) calls around existing phases.
//
// Phases come from two scopes:
//   * Orchestrator scope (CoreGPU::train, single-threaded between batches), gpuIndex = -1:
//       DataFetch, GpuTrain, GradMerge, WeightUpdate, KernelRestore
//   * Worker scope (CoreGPUWorker::trainSubset, one thread per GPU), gpuIndex >= 0:
//       H2DUpload, GpuCompute
//
// Note: in the no-BatchNorm fast path forward and backward propagation are a single
// fused OpenCL kernel queue executed by one run() per sample, so they are reported
// together as GpuCompute. Host->GPU writes are enqueued asynchronously (CL_FALSE), so
// the actual transfer is absorbed into the following GpuCompute (run() waits on it).
//===================================================================================================================//

namespace CNN
{
  enum class TimingPhase : int {
    DataFetch = 0,    // sampleProvider() — CPU, batch fetch / prefetch wait
    GpuTrain,         // fan-out of trainSubset across GPUs (wall clock, parallel)
    GradMerge,        // multi-GPU gradient merge on CPU (CNN + ANN)
    WeightUpdate,     // per-GPU weight update kernels
    KernelRestore,    // per-GPU training-kernel restore after update
    H2DUpload,        // host -> GPU writeBuffer enqueues (input + expected + dropout)
    GpuCompute,       // core->run() — fused forward + backward (+ accumulate)
    Count
  };

  enum class TimingEvent : int { Begin = 0, End = 1 };

  // (phase, event, gpuIndex). gpuIndex == -1 for orchestrator/main-thread phases.
  using TimingCallback = std::function<void(TimingPhase, TimingEvent, int)>;
}

//===================================================================================================================//

#endif // CNN_TIMINGCALLBACK_HPP
