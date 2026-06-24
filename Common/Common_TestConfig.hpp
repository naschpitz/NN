#ifndef COMMON_TESTCONFIG_HPP
#define COMMON_TESTCONFIG_HPP

#include <sys/types.h>

//===================================================================================================================//

namespace Common
{
  struct TestConfig {
      // Samples fetched/materialized per provider call during test/predict. Unlike
      // TrainConfig.batchSize, there is no optimizer update boundary in test — this
      // is purely a host-RAM streaming window, fanned out across workers/GPUs per call.
      ulong fetchSize = 64; // Fetch window for test/predict (default = 64)
  };
}

//===================================================================================================================//

#endif // COMMON_TESTCONFIG_HPP
