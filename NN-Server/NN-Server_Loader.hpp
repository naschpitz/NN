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

#include <json.hpp>

#include <string>
#include <utility>
#include <vector>

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

      //-- Package-aware loading --//

      // Check if the path refers to a .nnmodel package.
      static bool isPackage(const std::string& path);

      // Load JSON config + binary parameters from a .nnmodel package.
      static std::pair<nlohmann::json, std::vector<char>> loadPackage(const std::string& packagePath);

      // Load ANN configuration with pre-extracted binary parameters.
      // If binParams is empty, falls back to loadConfig(configFilePath).
      static ANN::CoreConfig<float> loadConfig(const std::string& configFilePath, const std::vector<char>& binParams);

      // Load CNN configuration with pre-extracted binary parameters.
      // If binParams is empty, falls back to loadCNNConfig(configFilePath).
      static CNN::CoreConfig<float> loadCNNConfig(const std::string& configFilePath,
                                                  const std::vector<char>& binParams);
  };

} // namespace NN_Server

#endif // NN_SERVER_LOADER_HPP
