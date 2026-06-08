#ifndef CNN_TIMINGCALLBACK_HPP
#define CNN_TIMINGCALLBACK_HPP

#include "Common/Common_Device.hpp"

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
  using namespace Common;
  enum class TimingPhase : int {
    // Orchestrator phases (gpuIndex = -1)
    DataFetch = 0,    // sampleProvider() — CPU, batch fetch / prefetch wait
    GpuTrain,         // fan-out of trainSubset across GPUs (wall clock, parallel)
    GradMerge,        // multi-GPU gradient merge on CPU (CNN + )
    WeightUpdate,     // per-GPU weight update kernels
    KernelRestore,    // per-GPU training-kernel restore after update
    // Worker phases (gpuIndex >= 0)
    H2DUpload,        // host -> GPU writeBuffer enqueues (input + expected + dropout)
    GpuCompute,       // core->run() — fused forward + backward (+ accumulate)
    // Sub-phase breakdown (populated from GPU profiling, not host-side events)
    Augmentation,     // GPU image augmentation (inside DataFetch, host-side)
    CnnForward,       // CNN propagate kernels (GPU-profiled)
    AnnForward,       //  propagate kernels (GPU-profiled)
    AnnBackward,      //  backprop kernels (GPU-profiled)
    CnnBackward,      // CNN backprop kernels (GPU-profiled)
    CNNAccumulate,    // CNN gradient accumulation (GPU-profiled)
    Accumulate,    //  gradient accumulation (GPU-profiled)
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
    // Kernel names are matched by prefix (the per-call suffix varies); a few categories need a
    // negative guard to disambiguate a forward kernel from its similarly-named backward variant.
    inline bool startsWith(const std::string& s, const char* prefix)
    {
      return s.rfind(prefix, 0) == 0;
    }

    inline bool contains(const std::string& s, const char* sub)
    {
      return s.find(sub) != std::string::npos;
    }

    inline TimingPhase categorizeKernel(const std::string& k)
    {
      // CNN forward: im2col (forward only), gemm_conv, relu, maxpool, avgpool,
      //              gap_propagate, gdp_propagate, norm_mean, norm_var,
      //              norm_normalize, res_add_proj, res_add, copy_cnn_to_ann
      if (startsWith(k, "im2col") && !contains(k, "_bk_"))
        return TimingPhase::CnnForward;

      if (startsWith(k, "gemm_conv"))
        return TimingPhase::CnnForward;

      if (startsWith(k, "relu") && !contains(k, "dRelu"))
        return TimingPhase::CnnForward;

      if (startsWith(k, "maxpool") && !contains(k, "dMaxpool"))
        return TimingPhase::CnnForward;

      if (startsWith(k, "avgpool") && !contains(k, "dAvgpool"))
        return TimingPhase::CnnForward;

      if (startsWith(k, "gap_propagate"))
        return TimingPhase::CnnForward;

      if (startsWith(k, "gdp_propagate"))
        return TimingPhase::CnnForward;

      if (startsWith(k, "norm_mean") || startsWith(k, "norm_var") ||
          (startsWith(k, "norm_normalize") && !contains(k, "bn_batch")))
        return TimingPhase::CnnForward;

      if (startsWith(k, "res_add"))
        return TimingPhase::CnnForward;

      if (startsWith(k, "copy_cnn_to_ann"))
        return TimingPhase::CnnForward;

      // CNN backward: im2col_bk_filt, gemm_dFilters, gemm_dInput, dBiases,
      //               col2im, zero_conv, zero_pool, dRelu, dMaxpool, dAvgpool,
      //               gap_back, gdp_back, norm_dGammaBeta, norm_dInput,
      //               res_bwd, res_bwd_proj, copy_ann_grad_to_cnn
      if (startsWith(k, "im2col_bk"))
        return TimingPhase::CnnBackward;

      if (startsWith(k, "gemm_dFilters") || startsWith(k, "gemm_dInput"))
        return TimingPhase::CnnBackward;

      if (startsWith(k, "dBiases"))
        return TimingPhase::CnnBackward;

      if (startsWith(k, "col2im"))
        return TimingPhase::CnnBackward;

      if (startsWith(k, "dRelu"))
        return TimingPhase::CnnBackward;

      if (startsWith(k, "dMaxpool"))
        return TimingPhase::CnnBackward;

      if (startsWith(k, "dAvgpool"))
        return TimingPhase::CnnBackward;

      if (startsWith(k, "gap_back"))
        return TimingPhase::CnnBackward;

      if (startsWith(k, "gdp_back"))
        return TimingPhase::CnnBackward;

      if (startsWith(k, "norm_dGammaBeta") || startsWith(k, "norm_dInput"))
        return TimingPhase::CnnBackward;

      if (startsWith(k, "zero_conv") || startsWith(k, "zero_pool"))
        return TimingPhase::CnnBackward;

      if (startsWith(k, "res_bwd"))
        return TimingPhase::CnnBackward;

      if (startsWith(k, "copy_ann_grad_to_cnn"))
        return TimingPhase::CnnBackward;

      // CNN accumulate: accum_filters, accum_biases, accum_norm_*, bn_accum_*
      if (startsWith(k, "accum_filters") || startsWith(k, "accum_biases") || startsWith(k, "accum_norm_") ||
          startsWith(k, "bn_accum_"))
        return TimingPhase::CNNAccumulate;

      //  forward: calculate_zs_layer, calculate_actvs_layer, apply_dropout_layer
      if (startsWith(k, "calculate_zs_layer") || startsWith(k, "calculate_actvs_layer") ||
          (startsWith(k, "apply_dropout_layer") && !contains(k, "backward")))
        return TimingPhase::AnnForward;

      //  backward: calculate_dCost_dActv_*, calculate_dCost_dBias_*, calculate_dCost_dWeight_*,
      //               apply_dropout_backward_layer*
      if (startsWith(k, "calculate_dCost_dActv") || startsWith(k, "calculate_dCost_dBias") ||
          startsWith(k, "calculate_dCost_dWeight") || startsWith(k, "apply_dropout_backward"))
        return TimingPhase::AnnBackward;

      //  accumulate: accumulate_dCost_dBiases, accumulate_dCost_dWeights
      if (startsWith(k, "accumulate_dCost_dBiases") || startsWith(k, "accumulate_dCost_dWeights"))
        return TimingPhase::Accumulate;

      // Loss
      if (startsWith(k, "calculate_sample_loss"))
        return TimingPhase::LossCompute;

      // BN cross-sample kernels (only in BN path)
      if (startsWith(k, "bn_batch") || startsWith(k, "bn_update_running"))
        return TimingPhase::CnnForward;

      // Update kernels (not profiled in sample loop, but categorize)
      if (startsWith(k, "update_parameters") || startsWith(k, "update_weights") || startsWith(k, "update_biases") ||
          startsWith(k, "norm_update_running"))
        return TimingPhase::WeightUpdate;

      return TimingPhase::CnnForward;
    }
  } // namespace Kernels

} // namespace CNN

//===================================================================================================================//

#endif // CNN_TIMINGCALLBACK_HPP
