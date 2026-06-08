#ifndef CNN_CORECONFIG_HPP
#define CNN_CORECONFIG_HPP

#include "Common/Common_Mode.hpp"
#include "Common/Common_Device.hpp"
#include "Common/Common_LogLevel.hpp"
#include "CNN_LayersConfig.hpp"
#include "Common/Common_CostFunctionConfig.hpp"
#include "Common/Common_TrainingConfig.hpp"
#include "Common/Common_TestConfig.hpp"
#include "CNN_Parameters.hpp"
#include "CNN_Types.hpp"

//===================================================================================================================//

namespace CNN
{
  using namespace Common;
  template <typename T>
  struct CoreConfig {
      Common::ModeType modeType;
      Common::DeviceType deviceType;
      int numThreads = 0; // 0 = use all available cores (for CPU mode)
      int numGPUs = 0; // 0 = use all available GPUs (for GPU mode)
      Shape3D inputShape; // Input tensor shape (C, H, W)
      LayersConfig layersConfig;
      Common::CostFunctionConfig<T> costFunctionConfig;
      Common::TrainingConfig<T> trainingConfig;
      Common::TestConfig testConfig;
      Parameters<T> parameters;
      ulong progressReports = 1000; // Number of progress reports (0 = no reports, default = 1000)
      Common::LogLevel logLevel = Common::LogLevel::ERROR;
  };
}

//===================================================================================================================//

#endif // CNN_CORECONFIG_HPP
