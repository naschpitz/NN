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
      // CNN forward: im2col (forward only), gemm_conv, relu, maxpool, avgpool,
      //              gap_propagate, gdp_propagate, norm_mean, norm_var,
      //              norm_normalize, res_add_proj, res_add, copy_cnn_to_ann
      if (kernelName.find("im2col") == 0 && kernelName.find("_bk_") == std::string::npos)
        return TimingPhase::CnnForward;

      if (kernelName.find("gemm_conv") == 0)
        return TimingPhase::CnnForward;

      if (kernelName.find("relu") == 0 && kernelName.find("dRelu") == std::string::npos)
        return TimingPhase::CnnForward;

      if (kernelName.find("maxpool") == 0 && kernelName.find("dMaxpool") == std::string::npos)
        return TimingPhase::CnnForward;

      if (kernelName.find("avgpool") == 0 && kernelName.find("dAvgpool") == std::string::npos)
        return TimingPhase::CnnForward;

      if (kernelName.find("gap_propagate") == 0)
        return TimingPhase::CnnForward;

      if (kernelName.find("gdp_propagate") == 0)
        return TimingPhase::CnnForward;

      if (kernelName.find("norm_mean") == 0 || kernelName.find("norm_var") == 0 ||
          (kernelName.find("norm_normalize") == 0 && kernelName.find("bn_batch") == std::string::npos))
        return TimingPhase::CnnForward;

      if (kernelName.find("res_add") == 0)
        return TimingPhase::CnnForward;

      if (kernelName.find("copy_cnn_to_ann") == 0)
        return TimingPhase::CnnForward;

      // CNN backward: im2col_bk_filt, gemm_dFilters, gemm_dInput, dBiases,
      //               col2im, zero_conv, zero_pool, dRelu, dMaxpool, dAvgpool,
      //               gap_back, gdp_back, norm_dGammaBeta, norm_dInput,
      //               res_bwd, res_bwd_proj, copy_ann_grad_to_cnn
      if (kernelName.find("im2col_bk") == 0)
        return TimingPhase::CnnBackward;

      if (kernelName.find("gemm_dFilters") == 0 || kernelName.find("gemm_dInput") == 0)
        return TimingPhase::CnnBackward;

      if (kernelName.find("dBiases") == 0)
        return TimingPhase::CnnBackward;

      if (kernelName.find("col2im") == 0)
        return TimingPhase::CnnBackward;

      if (kernelName.find("dRelu") == 0)
        return TimingPhase::CnnBackward;

      if (kernelName.find("dMaxpool") == 0)
        return TimingPhase::CnnBackward;

      if (kernelName.find("dAvgpool") == 0)
        return TimingPhase::CnnBackward;

      if (kernelName.find("gap_back") == 0)
        return TimingPhase::CnnBackward;

      if (kernelName.find("gdp_back") == 0)
        return TimingPhase::CnnBackward;

      if (kernelName.find("norm_dGammaBeta") == 0 || kernelName.find("norm_dInput") == 0)
        return TimingPhase::CnnBackward;

      if (kernelName.find("zero_conv") == 0 || kernelName.find("zero_pool") == 0)
        return TimingPhase::CnnBackward;

      if (kernelName.find("res_bwd") == 0)
        return TimingPhase::CnnBackward;

      if (kernelName.find("copy_ann_grad_to_cnn") == 0)
        return TimingPhase::CnnBackward;

      // CNN accumulate: accum_filters, accum_biases, accum_norm_*, bn_accum_*
      if (kernelName.find("accum_filters") == 0 || kernelName.find("accum_biases") == 0 ||
          kernelName.find("accum_norm_") == 0 || kernelName.find("bn_accum_") == 0)
        return TimingPhase::CNNAccumulate;

      // ANN forward: calculate_zs_layer, calculate_actvs_layer, apply_dropout_layer
      if (kernelName.find("calculate_zs_layer") == 0 || kernelName.find("calculate_actvs_layer") == 0 ||
          (kernelName.find("apply_dropout_layer") == 0 && kernelName.find("backward") == std::string::npos))
        return TimingPhase::AnnForward;

      // ANN backward: calculate_dCost_dActv_*, calculate_dCost_dBias_*, calculate_dCost_dWeight_*,
      //               apply_dropout_backward_layer*
      if (kernelName.find("calculate_dCost_dActv") == 0 || kernelName.find("calculate_dCost_dBias") == 0 ||
          kernelName.find("calculate_dCost_dWeight") == 0 || kernelName.find("apply_dropout_backward") == 0)
        return TimingPhase::AnnBackward;

      // ANN accumulate: accumulate_dCost_dBiases, accumulate_dCost_dWeights
      if (kernelName.find("accumulate_dCost_dBiases") == 0 || kernelName.find("accumulate_dCost_dWeights") == 0)
        return TimingPhase::ANNAccumulate;

      // Loss
      if (kernelName.find("calculate_sample_loss") == 0)
        return TimingPhase::LossCompute;

      // BN cross-sample kernels (only in BN path)
      if (kernelName.find("bn_batch") == 0 || kernelName.find("bn_update_running") == 0)
        return TimingPhase::CnnForward;

      // Update kernels (not profiled in sample loop, but categorize)
      if (kernelName.find("update_parameters") == 0 || kernelName.find("update_weights") == 0 ||
          kernelName.find("update_biases") == 0 || kernelName.find("norm_update_running") == 0)
        return TimingPhase::WeightUpdate;

      return TimingPhase::CnnForward;
    }
  } // namespace Kernels

} // namespace CNN

//===================================================================================================================//

#endif // CNN_TIMINGCALLBACK_HPP
