#ifndef ANN_CLI_LOADER_HPP
#define ANN_CLI_LOADER_HPP

#include <ANN_Core.hpp>
#include <ANN_Mode.hpp>
#include <ANN_Device.hpp>
#include <ANN_ActvFunc.hpp>
#include <ANN_LayersConfig.hpp>

#include <optional>
#include <string>

namespace ANN_CLI {

class Loader {
public:
  // Load configuration/model file.
  // The modeType parameter determines validation requirements:
  //   - TRAIN: requires layersConfig; trainingConfig and parameters are optional
  //            (parameters allow resuming training from a saved model)
  //   - RUN/TEST: requires layersConfig and parameters (trained weights/biases)
  //
  // If modeType or deviceType are not provided, values from the JSON config file are used.
  // If provided, they override the JSON config values.
  static ANN::CoreConfig<float> loadConfig(const std::string& configFilePath,
                                           std::optional<ANN::ModeType> modeType = std::nullopt,
                                           std::optional<ANN::DeviceType> deviceType = std::nullopt);

  static ANN::Samples<float> loadSamples(const std::string& samplesFilePath);

  static ANN::Input<float> loadInput(const std::string& inputFilePath);
};

} // namespace ANN_CLI

#endif // ANN_CLI_LOADER_HPP

