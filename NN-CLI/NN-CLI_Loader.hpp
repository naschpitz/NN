#ifndef NN_CLI_LOADER_HPP
#define NN_CLI_LOADER_HPP

#include "NN-CLI_NetworkType.hpp"
#include "NN-CLI_DataType.hpp"
#include "NN-CLI_IOConfig.hpp"

#include <ANN_Core.hpp>
#include <ANN_Mode.hpp>
#include <ANN_Device.hpp>
#include <ANN_ActvFunc.hpp>
#include <ANN_LayersConfig.hpp>

#include <CNN_Core.hpp>
#include <CNN_CoreConfig.hpp>
#include <CNN_Mode.hpp>
#include <CNN_Device.hpp>
#include <CNN_LayersConfig.hpp>
#include <CNN_SlidingStrategy.hpp>
#include <CNN_PoolType.hpp>

#include <optional>
#include <string>

namespace NN_CLI {

class Loader {
public:
  // Detect whether a config file defines an ANN or CNN network.
  static NetworkType detectNetworkType(const std::string& configFilePath);

  // Load I/O configuration (inputType, outputType, shapes) with optional CLI overrides
  static IOConfig loadIOConfig(const std::string& configFilePath,
                                std::optional<std::string> inputTypeOverride  = std::nullopt,
                                std::optional<std::string> outputTypeOverride = std::nullopt);

  // Load ANN configuration with optional CLI overrides
  static ANN::CoreConfig<float> loadANNConfig(const std::string& configFilePath,
                                               std::optional<ANN::ModeType> modeType = std::nullopt,
                                               std::optional<ANN::DeviceType> deviceType = std::nullopt);

  // Load CNN configuration with optional CLI overrides
  static CNN::CoreConfig<float> loadCNNConfig(const std::string& configFilePath,
                                               std::optional<std::string> modeOverride = std::nullopt,
                                               std::optional<std::string> deviceOverride = std::nullopt);

  // Load ANN samples from JSON (supports image paths when ioConfig.inputType/outputType is IMAGE)
  static ANN::Samples<float> loadANNSamples(const std::string& samplesFilePath,
                                             const IOConfig& ioConfig);

  // Load CNN samples from JSON (supports image paths when ioConfig.inputType/outputType is IMAGE)
  static CNN::Samples<float> loadCNNSamples(const std::string& samplesFilePath,
                                             const CNN::Shape3D& inputShape,
                                             const IOConfig& ioConfig);

  // Load ANN inputs from JSON (batch: "inputs" array; supports image paths when ioConfig.inputType is IMAGE)
  static std::vector<ANN::Input<float>> loadANNInputs(const std::string& inputFilePath,
                                                       const IOConfig& ioConfig);

  // Load CNN inputs from JSON (batch: "inputs" array; supports image paths when ioConfig.inputType is IMAGE)
  static std::vector<CNN::Input<float>> loadCNNInputs(const std::string& inputFilePath,
                                                       const CNN::Shape3D& inputShape,
                                                       const IOConfig& ioConfig);
};

} // namespace NN_CLI

#endif // NN_CLI_LOADER_HPP

