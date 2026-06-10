#ifndef COMMON_TRAININGPROGRESSEVENT_HPP
#define COMMON_TRAININGPROGRESSEVENT_HPP

#include <functional>
#include <sys/types.h>

#include "Common/Common_EpochCompletionEvent.hpp"

//===================================================================================================================//

namespace Common
{
  // Training progress information passed to callbacks
  template <typename T>
  struct TrainingProgressEvent {
      ulong currentEpoch;
      ulong totalEpochs;
      ulong currentSample;
      ulong totalSamples;
      T epochLoss; // Average loss for completed epoch (0 if epoch not complete)
      T sampleLoss; // Loss for current sample

      // Monitoring signals
      bool isNewBest = false; // Monitor found a new best loss — caller should save
      bool stoppedEarly = false; // Monitor decided to stop training

      // Multi-GPU progress tracking
      int gpuIndex = -1; // -1 = not GPU-specific (epoch completion, CPU mode), >= 0 = specific GPU
      int totalGPUs = 0; // Total number of GPUs being used
  };

  // Callback type for training progress
  template <typename T>
  using TrainingCallback = std::function<void(const TrainingProgressEvent<T>&)>;
}

//===================================================================================================================//

#endif // COMMON_TRAININGPROGRESSEVENT_HPP
