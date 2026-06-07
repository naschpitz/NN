#ifndef NN_CLI_GPUAUGMENTER_HPP
#define NN_CLI_GPUAUGMENTER_HPP

#include "NN-CLI_AugmentationTransforms.hpp"
#include "NN-CLI_LogLevel.hpp"
#include "NN-CLI_Types.hpp"

#include <OCLW_Core.hpp>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <random>
#include <vector>

namespace NN_CLI
{
  // Runs image augmentation on a single GPU via OpenCL. Operates on a contiguous
  // batch of NCHW float images in [0, 1] (count * C * H * W), modifying it in place.
  // Per-sample random parameters are drawn on the host (matching the CPU transforms'
  // distributions) and uploaded; the kernels apply them on the GPU. Not internally
  // synchronised — GpuAugmenterPool guarantees exclusive use per instance.
  class GpuAugmenter
  {
    public:
      GpuAugmenter(int deviceIndex, ulong c, ulong h, ulong w, LogLevel logLevel);

      void augment(std::vector<float>& batch, ulong count, const AugmentationTransforms& transforms, float probability,
                   std::mt19937& rng);

    private:
      OpenCLWrapper::Core core;
      ulong C;
      ulong H;
      ulong W;
      LogLevel logLevel;
      ulong capacity = 0;
      bool sourcesLoaded = false;

      //-- Host-side per-sample parameter staging buffers (sized to capacity) --//
      std::vector<int> applyFlip, applyRot, applyTrans, applyScale, applyElastic;
      std::vector<int> applyHue, applyBri, applyCon, applyErase, applyNoise;
      std::vector<float> angleRad, scales, shiftDeg, deltas, factors;
      std::vector<int> dxs, dys, eraseX0, eraseY0, eraseW, eraseH;
  };

  // A pool of GpuAugmenters, one per GPU device. augment() runs on whichever GPU is
  // free (blocking until one is available), so augmentation load spreads across GPUs
  // and is safe to call from multiple prefetch worker threads.
  class GpuAugmenterPool
  {
    public:
      GpuAugmenterPool(const std::vector<int>& deviceIndices, ulong c, ulong h, ulong w, LogLevel logLevel);

      void augment(std::vector<float>& batch, ulong count, const AugmentationTransforms& transforms, float probability);

      void setTimingCallback(std::function<void(bool)> callback)
      {
        timingCallback = std::move(callback);
      }

      bool empty() const
      {
        return this->augmenters.empty();
      }

    private:
      std::vector<std::unique_ptr<GpuAugmenter>> augmenters;
      std::vector<std::unique_ptr<std::mt19937>> rngs;
      std::vector<int> freeList;
      std::mutex mutex;
      std::condition_variable cv;
      std::function<void(bool)> timingCallback;
  };
}

#endif // NN_CLI_GPUAUGMENTER_HPP
