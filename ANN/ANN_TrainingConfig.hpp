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
    bool shuffleSamples = true;   // Shuffle sample order each epoch (default = true)
    float dropoutRate = 0.0f;     // Dropout probability for hidden layers (0.0 = disabled)
  };
}

//===================================================================================================================//

#endif // ANN_TRAININGCONFIG_HPP

