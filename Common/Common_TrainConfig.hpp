#ifndef COMMON_TRAINCONFIG_HPP
#define COMMON_TRAINCONFIG_HPP

#include "Common_LearningRateScheduler.hpp"
#include "Common_MonitoringConfig.hpp"
#include "Common_Optimizer.hpp"
#include "Common_ValidationConfig.hpp"

#include <cstdint>
#include <sys/types.h>

//===================================================================================================================//

namespace Common
{
  template <typename T>
  struct TrainConfig {
      ulong numEpochs = 0;
      ulong startingEpoch = 0; // Epoch index to start training from (0 = fresh start; >0 = resume)
      float learningRate = 0.01f;
      ulong batchSize = 64; // Mini-batch size (default = 64)
      // Samples to fetch/materialize per SampleProvider call (host RAM window).
      // 0 (default) = auto: max(numWorkers, augmenterChunkSize) for InstanceNorm,
      // batchSize for BatchNorm. Smaller values reduce peak host RAM at the cost of
      // more frequent I/O. See todo/decouple-fetchsize-from-batchsize.md.
      ulong fetchSize = 0;
      bool shuffleSamples = true; // Shuffle sample order each epoch (default = true)
      // RNG seed for the per-epoch shuffle. 0 (default) means
      // std::random_device — non-deterministic, what production wants. Tests
      // set a non-zero value so the whole training run reproduces.
      uint32_t shuffleSeed = 0;
      float dropoutRate = 0.0f; // Dropout probability for dense hidden layers (0.0 = disabled)
      LearningRateSchedulerConfig
        learningRateScheduler; // LR scheduler (default: NONE = flat LR, behavior identical to today)
      Optimizer<T> optimizer; // Optimizer (default: SGD)
      ValidationConfig validationDataset; // Validation split config (default: enabled, auto-size)
      MonitoringConfig monitoringConfig; // Training health monitoring (default: disabled)
  };
}

//===================================================================================================================//

#endif // COMMON_TRAINCONFIG_HPP
