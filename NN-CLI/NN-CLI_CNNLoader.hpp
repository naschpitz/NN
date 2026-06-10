#ifndef NN_CLI_CNNLOADER_HPP
#define NN_CLI_CNNLOADER_HPP

#include "NN-CLI_IOConfig.hpp"

#include <CNN_Core.hpp>
#include <CNN_CoreConfig.hpp>
#include "Common/Common_Mode.hpp"
#include "Common/Common_Device.hpp"
#include <CNN_LayersConfig.hpp>
#include <CNN_SlidingStrategy.hpp>
#include <CNN_PoolType.hpp>

#include <json.hpp>

#include <optional>
#include <string>
#include <vector>

namespace NN_CLI
{

  class CNNLoader
  {
    public:
      // Load CNN configuration with optional CLI overrides (file-path convenience wrapper).
      static CNN::CoreConfig<float> loadConfig(const std::string& configFilePath,
                                               std::optional<std::string> modeOverride = std::nullopt,
                                               std::optional<std::string> deviceOverride = std::nullopt);

      // Load CNN configuration from pre-parsed JSON.
      static CNN::CoreConfig<float> loadConfig(const nlohmann::json& json,
                                               std::optional<std::string> modeOverride = std::nullopt,
                                               std::optional<std::string> deviceOverride = std::nullopt);

      // Load CNN configuration from pre-parsed JSON with binary parameters (for .nnmodel packages).
      static CNN::CoreConfig<float> loadConfig(const nlohmann::json& json, const std::vector<char>& binParams,
                                               std::optional<std::string> modeOverride = std::nullopt,
                                               std::optional<std::string> deviceOverride = std::nullopt);

      // Load CNN samples from JSON (supports image paths when ioConfig.inputType/outputType is IMAGE)
      static CNN::Samples<float> loadSamples(const std::string& samplesFilePath, const CNN::Shape3D& inputShape,
                                             const IOConfig& ioConfig, ulong progressReports = 1000);

      // Load CNN inputs from JSON (batch: "inputs" array; supports image paths when ioConfig.inputType is IMAGE)
      static std::vector<CNN::Input<float>> loadInputs(const std::string& inputFilePath, const CNN::Shape3D& inputShape,
                                                       const IOConfig& ioConfig, ulong progressReports = 1000);
  };

} // namespace NN_CLI

#endif // NN_CLI_CNNLOADER_HPP
