#ifndef NN_CLI_LOADER_HPP
#define NN_CLI_LOADER_HPP

#include "NN-CLI_AugmentationConfig.hpp"
#include "NN-CLI_NetworkType.hpp"
#include "NN-CLI_IOConfig.hpp"

#include <json.hpp>

#include <optional>
#include <string>

namespace NN_CLI
{

  class Loader
  {
    public:
      // Parse a config file once and return the JSON object for reuse across multiple loader calls.
      static nlohmann::json parseConfigFile(const std::string& configFilePath);

      // Detect whether a config file defines an  or CNN network (file-path convenience wrapper).
      static NetworkType detectNetworkType(const std::string& configFilePath);

      // Detect whether a config file defines an  or CNN network from pre-parsed JSON.
      static NetworkType detectNetworkType(const nlohmann::json& json);

      // Load I/O configuration (inputType, outputType, shapes) with optional CLI overrides (file-path wrapper).
      static IOConfig loadIOConfig(const std::string& configFilePath,
                                   std::optional<std::string> inputTypeOverride = std::nullopt,
                                   std::optional<std::string> outputTypeOverride = std::nullopt);

      // Load I/O configuration from pre-parsed JSON.
      static IOConfig loadIOConfig(const nlohmann::json& json,
                                   std::optional<std::string> inputTypeOverride = std::nullopt,
                                   std::optional<std::string> outputTypeOverride = std::nullopt);

      // Load progressReports from config root (returns 1000 if not present) (file-path wrapper).
      static ulong loadProgressReports(const std::string& configFilePath);

      // Load progressReports from pre-parsed JSON.
      static ulong loadProgressReports(const nlohmann::json& json);

      // Load saveModelInterval from config root (returns 10 if not present; 0 = disabled) (file-path wrapper).
      static ulong loadSaveModelInterval(const std::string& configFilePath);

      // Load saveModelInterval from pre-parsed JSON.
      static ulong loadSaveModelInterval(const nlohmann::json& json);

      // Load data augmentation config from trainingConfig (file-path wrapper).
      static AugmentationConfig loadAugmentationConfig(const std::string& configFilePath);

      // Load data augmentation config from pre-parsed JSON.
      static AugmentationConfig loadAugmentationConfig(const nlohmann::json& json);
  };

} // namespace NN_CLI

#endif // NN_CLI_LOADER_HPP
