#ifndef NN_CLI_ANNLOADER_HPP
#define NN_CLI_ANNLOADER_HPP

#include "NN-CLI_IOConfig.hpp"

#include <_Core.hpp>
#include "Common/Common_Mode.hpp"
#include "Common/Common_Device.hpp"
#include <_ActvFunc.hpp>
#include <_LayersConfig.hpp>

#include <json.hpp>

#include <optional>
#include <string>

namespace NN_CLI
{

  class ANNLoader
  {
    public:
      // Load  configuration with optional CLI overrides (file-path convenience wrapper).
      static ANN::CoreConfig<float> loadConfig(const std::string& configFilePath,
                                               std::optional<Common::ModeType> modeType = std::nullopt,
                                               std::optional<Common::DeviceType> deviceType = std::nullopt);

      // Load  configuration from pre-parsed JSON.
      static ANN::CoreConfig<float> loadConfig(const nlohmann::json& json,
                                               std::optional<Common::ModeType> modeType = std::nullopt,
                                               std::optional<Common::DeviceType> deviceType = std::nullopt);

      // Load  samples from JSON (supports image paths when ioConfig.inputType/outputType is IMAGE)
      static ANN::Samples<float> loadSamples(const std::string& samplesFilePath, const IOConfig& ioConfig,
                                             ulong progressReports = 1000);

      // Load  inputs from JSON (batch: "inputs" array; supports image paths when ioConfig.inputType is IMAGE)
      static std::vector<ANN::Input<float>> loadInputs(const std::string& inputFilePath, const IOConfig& ioConfig,
                                                       ulong progressReports = 1000);
  };

} // namespace NN_CLI

#endif // NN_CLI_LOADER_HPP
