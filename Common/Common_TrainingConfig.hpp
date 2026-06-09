#ifndef COMMON_TRAININGCONFIG_HPP
#define COMMON_TRAININGCONFIG_HPP

#include "Common_MonitoringConfig.hpp"
#include "Common_Optimizer.hpp"
#include "Common_ValidationConfig.hpp"

#include <cstdint>
#include <sys/types.h>

//===================================================================================================================//

namespace Common
{
  template <typename T>
  struct TrainingConfig {
      ulong numEpochs = 0;
      ulong startingEpoch = 0; // Epoch index to start training from (0 = fresh start; >0 = resume)
      float learningRate = 0.01f;
      ulong batchSize = 64; // Mini-batch size (default = 64)
      bool shuffleSamples = true; // Shuffle sample order each epoch (default = true)
      // RNG seed for the per-epoch shuffle. 0 (default) means
      // std::random_device — non-deterministic, what production wants. Tests
      // set a non-zero value so the whole training run reproduces.
      uint32_t shuffleSeed = 0;
      float dropoutRate = 0.0f; // Dropout probability for dense hidden layers (0.0 = disabled)
      Optimizer<T> optimizer; // Optimizer (default: SGD)
      ValidationConfig validationDataset; // Validation split config (default: enabled, auto-size)
      MonitoringConfig monitoringConfig; // Training health monitoring (default: disabled)
  };
}

//===================================================================================================================//

#endif // COMMON_TRAININGCONFIG_HPP
