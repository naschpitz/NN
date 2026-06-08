#ifndef NN_SERVER_LOADER_HPP
#define NN_SERVER_LOADER_HPP

#include "NN-Server_IOConfig.hpp"
#include "NN-Server_NetworkType.hpp"

#include <ANN_Core.hpp>
#include "Common/Common_Mode.hpp"
#include "Common/Common_Device.hpp"
#include <ANN_ActvFunc.hpp>
#include <ANN_LayersConfig.hpp>

#include <CNN_Core.hpp>
#include <CNN_CoreConfig.hpp>
#include "Common/Common_Mode.hpp"
#include "Common/Common_Device.hpp"
#include <CNN_LayersConfig.hpp>
#include <CNN_SlidingStrategy.hpp>
#include <CNN_PoolType.hpp>

#include <string>

namespace NN_Server
{

  class Loader
  {
    public:
      // Detect whether a config file defines an  or CNN network.
      static NetworkType detectNetworkType(const std::string& configFilePath);

      // Load input configuration from the model file (inputType + inputShape).
      static InputConfig loadInputConfig(const std::string& configFilePath);

      // Load output configuration from the model file (outputType + outputShape).
      static OutputConfig loadOutputConfig(const std::string& configFilePath);

      // Load ANN configuration (always in PREDICT mode).
      static ANN::CoreConfig<float> loadConfig(const std::string& configFilePath);

      // Load CNN configuration (always in PREDICT mode).
      static CNN::CoreConfig<float> loadCNNConfig(const std::string& configFilePath);
  };

} // namespace NN_Server

#endif // NN_SERVER_LOADER_HPP
