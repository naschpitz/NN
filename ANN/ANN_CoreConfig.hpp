#ifndef _CORECONFIG_HPP
#define _CORECONFIG_HPP

#include "Common/Common_Mode.hpp"
#include "Common/Common_Device.hpp"
#include "Common/Common_LogLevel.hpp"
#include "_LayersConfig.hpp"
#include "Common/Common_CostFunctionConfig.hpp"
#include "Common/Common_TrainingConfig.hpp"
#include "Common/Common_TestConfig.hpp"
#include "_Parameters.hpp"

//===================================================================================================================//

namespace ANN
{
  using namespace Common;
  template <typename T>
  struct CoreConfig {
      ModeType modeType;
      DeviceType deviceType;
      int numThreads = 0; // 0 = use all available cores (for CPU mode)
      int numGPUs = 0; // 0 = use all available GPUs (for GPU mode)
      LayersConfig layersConfig;
      CostFunctionConfig<T> costFunctionConfig;
      TrainingConfig<T> trainingConfig;
      TestConfig testConfig;
      Parameters<T> parameters;
      ulong progressReports = 1000; // Number of progress reports (0 = no reports, default = 1000)
      LogLevel logLevel = LogLevel::ERROR;
  };
}

//===================================================================================================================//

#endif // _CORECONFIG_HPP
