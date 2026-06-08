#ifndef NN_CLI_ANNLOADER_HPP
#define NN_CLI_ANNLOADER_HPP

#include "NN-CLI_IOConfig.hpp"

#include <ANN_Core.hpp>
#include <ANN_Mode.hpp>
#include <ANN_Device.hpp>
#include <ANN_ActvFunc.hpp>
#include <ANN_LayersConfig.hpp>

#include <json.hpp>

#include <optional>
#include <string>

namespace NN_CLI
{

  class ANNLoader
  {
    public:
      // Load ANN configuration with optional CLI overrides (file-path convenience wrapper).
      static ANN::CoreConfig<float> loadConfig(const std::string& configFilePath,
                                               std::optional<ANN::ModeType> modeType = std::nullopt,
                                               std::optional<ANN::DeviceType> deviceType = std::nullopt);

      // Load ANN configuration from pre-parsed JSON.
      static ANN::CoreConfig<float> loadConfig(const nlohmann::json& json,
                                               std::optional<ANN::ModeType> modeType = std::nullopt,
                                               std::optional<ANN::DeviceType> deviceType = std::nullopt);

      // Load ANN samples from JSON (supports image paths when ioConfig.inputType/outputType is IMAGE)
      static ANN::Samples<float> loadSamples(const std::string& samplesFilePath, const IOConfig& ioConfig,
                                             ulong progressReports = 1000);

      // Load ANN inputs from JSON (batch: "inputs" array; supports image paths when ioConfig.inputType is IMAGE)
      static std::vector<ANN::Input<float>> loadInputs(const std::string& inputFilePath, const IOConfig& ioConfig,
                                                       ulong progressReports = 1000);
  };

} // namespace NN_CLI

#endif // NN_CLI_ANNLOADER_HPP
