#ifndef ANN_TRAININGCONFIG_HPP
#define ANN_TRAININGCONFIG_HPP

#include <sys/types.h>

//===================================================================================================================//

namespace ANN {
  template <typename T>
  struct TrainingConfig {
    ulong numEpochs = 0;
    float learningRate = 0.01f;
    ulong batchSize = 64;         // Mini-batch size (default = 64)
  };
}

//===================================================================================================================//

#endif // ANN_TRAININGCONFIG_HPP

