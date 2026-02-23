#ifndef ANN_CORECONFIG_HPP
#define ANN_CORECONFIG_HPP

#include "ANN_Mode.hpp"
#include "ANN_Device.hpp"
#include "ANN_LogLevel.hpp"
#include "ANN_LayersConfig.hpp"
#include "ANN_CostFunctionConfig.hpp"
#include "ANN_TrainingConfig.hpp"
#include "ANN_Parameters.hpp"

//===================================================================================================================//

namespace ANN {
  template <typename T>
  struct CoreConfig {
    ModeType modeType;
    DeviceType deviceType;
    LayersConfig layersConfig;
    CostFunctionConfig<T> costFunctionConfig;
    TrainingConfig<T> trainingConfig;
    Parameters<T> parameters;
    ulong progressReports = 1000; // Number of progress reports (0 = no reports, default = 1000)
    LogLevel logLevel = LogLevel::ERROR;
  };
}

//===================================================================================================================//

#endif // ANN_CORECONFIG_HPP

