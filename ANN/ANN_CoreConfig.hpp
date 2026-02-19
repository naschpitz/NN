#ifndef ANN_CORECONFIG_HPP
#define ANN_CORECONFIG_HPP

#include "ANN_Mode.hpp"
#include "ANN_Device.hpp"
#include "ANN_LayersConfig.hpp"
#include "ANN_TrainingConfig.hpp"
#include "ANN_Parameters.hpp"

//===================================================================================================================//

namespace ANN {
  template <typename T>
  struct CoreConfig {
    ModeType modeType;
    DeviceType deviceType;
    LayersConfig layersConfig;
    TrainingConfig<T> trainingConfig;
    Parameters<T> parameters;
    bool verbose = false;
  };
}

//===================================================================================================================//

#endif // ANN_CORECONFIG_HPP

