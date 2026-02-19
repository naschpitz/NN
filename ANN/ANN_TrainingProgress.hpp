#ifndef ANN_TRAININGPROGRESS_HPP
#define ANN_TRAININGPROGRESS_HPP

#include <functional>
#include <sys/types.h>

//===================================================================================================================//

namespace ANN {
  // Training progress information passed to callbacks
  template <typename T>
  struct TrainingProgress {
    ulong currentEpoch;
    ulong totalEpochs;
    ulong currentSample;
    ulong totalSamples;
    T epochLoss;        // Average loss for completed epoch (0 if epoch not complete)
    T sampleLoss;       // Loss for current sample

    // Multi-GPU progress tracking
    int gpuIndex = -1;  // -1 = not GPU-specific (epoch completion, CPU mode), >= 0 = specific GPU
    int totalGPUs = 1;  // Total number of GPUs being used
  };

  // Callback type for training progress
  template <typename T>
  using TrainingCallback = std::function<void(const TrainingProgress<T>&)>;
}

//===================================================================================================================//

#endif // ANN_TRAININGPROGRESS_HPP

