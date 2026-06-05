#ifndef CNN_TIMINGCALLBACK_HPP
#define CNN_TIMINGCALLBACK_HPP

#include <functional>
#include <string>
#include <vector>

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
// In addition to host-side wall-clock timing (TimingCallback), the library provides
// GPU-internal per-kernel profiling via OpenCL's CL_QUEUE_PROFILING_ENABLE. After each
// core->run() in the fast path, kernel execution times are collected from the device,
// categorized into logical sub-phases, and reported via GpuProfileCallback.
//
// Note: in the no-BatchNorm fast path forward and backward propagation are a single
// fused OpenCL kernel queue executed by one run() per sample, so they are reported
// together as GpuCompute. Host->GPU writes are enqueued asynchronously (CL_FALSE), so
// the actual transfer is absorbed into the following GpuCompute (run() waits on it).
//===================================================================================================================//

namespace CNN
{
  enum class TimingPhase : int {
    // Orchestrator phases (gpuIndex = -1)
    DataFetch = 0,    // sampleProvider() — CPU, batch fetch / prefetch wait
    GpuTrain,         // fan-out of trainSubset across GPUs (wall clock, parallel)
    GradMerge,        // multi-GPU gradient merge on CPU (CNN + ANN)
    WeightUpdate,     // per-GPU weight update kernels
    KernelRestore,    // per-GPU training-kernel restore after update
    // Worker phases (gpuIndex >= 0)
    H2DUpload,        // host -> GPU writeBuffer enqueues (input + expected + dropout)
    GpuCompute,       // core->run() — fused forward + backward (+ accumulate)
    // Sub-phase breakdown (populated from GPU profiling, not host-side events)
    Augmentation,     // GPU image augmentation (inside DataFetch, host-side)
    CnnForward,       // CNN propagate kernels (GPU-profiled)
    AnnForward,       // ANN propagate kernels (GPU-profiled)
    AnnBackward,      // ANN backprop kernels (GPU-profiled)
    CnnBackward,      // CNN backprop kernels (GPU-profiled)
    CNNAccumulate,    // CNN gradient accumulation (GPU-profiled)
    ANNAccumulate,    // ANN gradient accumulation (GPU-profiled)
    LossCompute,      // loss kernel (GPU-profiled)
    Count
  };

  enum class TimingEvent : int { Begin = 0, End = 1 };

  // (phase, event, gpuIndex). gpuIndex == -1 for orchestrator/main-thread phases.
  using TimingCallback = std::function<void(TimingPhase, TimingEvent, int)>;

  //===================================================================================================================//
  //-- GPU profile data --//
  //===================================================================================================================//

  struct GpuPhaseProfile
  {
      TimingPhase phase;   // sub-phase category
      double gpuMs;        // GPU-measured execution time (sum of kernel times in this phase)
      ulong kernelCalls;   // number of kernel enqueues in this phase
  };

  // Invoked after each core->run() in the fast path with per-sub-phase GPU-measured
  // execution times and the GPU index. The callee receives a snapshot; the worker
  // resets profiling accumulators before the next sample.
  using GpuProfileCallback = std::function<void(const std::vector<GpuPhaseProfile>&, int gpuIndex)>;

  //===================================================================================================================//
  //-- Kernel-to-phase categorization --//
  //===================================================================================================================//

  namespace Kernels
  {
    inline TimingPhase categorizeKernel(const std::string& kernelName)
    {
      // CNN forward kernels
      if (kernelName.find("im2col") == 0 || kernelName.find("gemm") == 0 || kernelName.find("col2im") == 0)
        return TimingPhase::CnnForward;

      if (kernelName.find("calculate_relu") == 0 || kernelName == "calculate_actvs_from_relu" ||
          kernelName.find("calculate_maxpool") == 0 || kernelName.find("calculate_avgpool") == 0 ||
          kernelName.find("gap_propagate") == 0 || kernelName.find("gdp_propagate") == 0 ||
          kernelName.find("norm_compute") == 0 || kernelName.find("norm_normalize") == 0 ||
          kernelName.find("residual_add") == 0)
        return TimingPhase::CnnForward;

      // CNN backward kernels
      if (kernelName.find("calculate_dCost_dRelu") == 0 || kernelName.find("calculate_dCost_dMaxpool") == 0 ||
          kernelName.find("calculate_dCost_dAvgpool") == 0 || kernelName.find("gap_backpropagate") == 0 ||
          kernelName.find("gdp_backpropagate") == 0 || kernelName.find("norm_dGamma") == 0 ||
          kernelName.find("norm_dBeta") == 0 || kernelName.find("norm_dInput") == 0 ||
          kernelName.find("calculate_dCost_dBiases_cnn") == 0 || kernelName.find("norm_") == 0 ||
          kernelName.find("residual_bwd") == 0)
        return TimingPhase::CnnBackward;

      // Bridge
      if (kernelName.find("copy_cnn_to_ann") == 0)
        return TimingPhase::CnnForward; // lumped with CNN forward

      if (kernelName.find("copy_ann_to_cnn") == 0)
        return TimingPhase::CnnBackward; // lumped with CNN backward

      // ANN forward kernels
      if (kernelName.find("calculate_zs") == 0 || kernelName.find("calculate_actvs") == 0 ||
          kernelName.find("apply_dropout") == 0)
        return TimingPhase::AnnForward;

      // ANN backward kernels
      if (kernelName.find("calculate_dCost_dActv") == 0 || kernelName.find("calculate_dCost_dBias") == 0 ||
          kernelName.find("calculate_dCost_dWeight") == 0 || kernelName.find("apply_dropout_backward") == 0)
        return TimingPhase::AnnBackward;

      // Accumulate kernels
      if (kernelName.find("accumulate_dCost") == 0 || kernelName.find("accumulate_gradients") == 0) {
        if (kernelName.find("cnn_accum") != std::string::npos || kernelName.find("cnn_dFilters") != std::string::npos ||
            kernelName.find("cnn_dBiases") != std::string::npos ||
            kernelName.find("cnn_norm_dGamma") != std::string::npos ||
            kernelName.find("cnn_norm_dBeta") != std::string::npos)
          return TimingPhase::CNNAccumulate;

        return TimingPhase::ANNAccumulate;
      }

      // Loss kernel
      if (kernelName.find("calculate_sample_loss") == 0)
        return TimingPhase::LossCompute;

      // BatchNorm kernels (used in BN path, but categorize for completeness)
      if (kernelName.find("bn_accum") == 0)
        return TimingPhase::CNNAccumulate;

      // Update kernels (not profiled in the fast path sample loop, but categorize anyway)
      if (kernelName.find("update_parameters") == 0 || kernelName.find("update_weights") == 0 ||
          kernelName.find("update_biases") == 0)
        return TimingPhase::WeightUpdate;

      // Default: treat as CNN forward (most kernels are in the forward path)
      return TimingPhase::CnnForward;
    }
  } // namespace Kernels

} // namespace CNN

//===================================================================================================================//

#endif // CNN_TIMINGCALLBACK_HPP
