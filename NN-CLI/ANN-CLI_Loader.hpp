#ifndef ANN_CLI_LOADER_HPP
#define ANN_CLI_LOADER_HPP

#include <ANN_Core.hpp>
#include <ANN_CoreMode.hpp>
#include <ANN_CoreType.hpp>
#include <ANN_ActvFunc.hpp>
#include <ANN_LayersConfig.hpp>

#include <optional>
#include <string>

namespace ANN_CLI {

class Loader {
public:
  // Load config file for training (architecture + training settings, no weights)
  // If modeType or deviceType are not provided, values from the JSON config file are used.
  // If provided, they override the JSON config values.
  static ANN::CoreConfig<float> loadConfig(const std::string& configFilePath,
                                           std::optional<ANN::CoreModeType> modeType = std::nullopt,
                                           std::optional<ANN::DeviceType> deviceType = std::nullopt);

  // Load trained model file for run/test modes (architecture + trained weights)
  // If deviceType is not provided, value from the JSON config file is used.
  // If provided, it overrides the JSON config value.
  static ANN::CoreConfig<float> loadModel(const std::string& modelFilePath,
                                          std::optional<ANN::DeviceType> deviceType = std::nullopt);

  static ANN::Samples<float> loadSamples(const std::string& samplesFilePath);

  static ANN::Input<float> loadInput(const std::string& inputFilePath);
};

} // namespace ANN_CLI

#endif // ANN_CLI_LOADER_HPP

