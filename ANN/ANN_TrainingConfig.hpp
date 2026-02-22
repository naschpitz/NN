#ifndef ANN_TRAININGCONFIG_HPP
#define ANN_TRAININGCONFIG_HPP

#include <sys/types.h>

//===================================================================================================================//

namespace ANN {
  template <typename T>
  struct TrainingConfig {
    ulong numEpochs = 0;
    float learningRate = 0.01f;
    int numThreads = 0;           // 0 = use all available cores (for CPU mode)
    int numGPUs = 0;              // 0 = use all available GPUs, 1 = single GPU (default behavior)
  };
}

//===================================================================================================================//

#endif // ANN_TRAININGCONFIG_HPP

