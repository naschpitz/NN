#ifndef NN_CLI_LOADER_HPP
#define NN_CLI_LOADER_HPP

#include "NN-CLI_AugmentationConfig.hpp"
#include "NN-CLI_NetworkType.hpp"
#include "NN-CLI_IOConfig.hpp"

#include <optional>
#include <string>

namespace NN_CLI
{

  class Loader
  {
    public:
      // Detect whether a config file defines an ANN or CNN network.
      static NetworkType detectNetworkType(const std::string& configFilePath);

      // Load I/O configuration (inputType, outputType, shapes) with optional CLI overrides
      static IOConfig loadIOConfig(const std::string& configFilePath,
                                   std::optional<std::string> inputTypeOverride = std::nullopt,
                                   std::optional<std::string> outputTypeOverride = std::nullopt);

      // Load progressReports from config root (returns 1000 if not present)
      static ulong loadProgressReports(const std::string& configFilePath);

      // Load saveModelInterval from config root (returns 10 if not present; 0 = disabled)
      static ulong loadSaveModelInterval(const std::string& configFilePath);

      // Load data augmentation config from trainingConfig (NN-CLI handles augmentation, not ANN/CNN)
      static AugmentationConfig loadAugmentationConfig(const std::string& configFilePath);
  };

} // namespace NN_CLI

#endif // NN_CLI_LOADER_HPP
