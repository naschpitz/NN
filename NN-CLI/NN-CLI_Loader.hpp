#ifndef NN_CLI_LOADER_HPP
#define NN_CLI_LOADER_HPP

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

enum class NetworkType { ANN, CNN };

class Loader {
public:
  // Detect whether a config file defines an ANN or CNN network.
  // CNN configs contain "inputShape" and/or "cnnLayersConfig".
  // ANN configs contain "layersConfig" with "numNeurons".
  static NetworkType detectNetworkType(const std::string& configFilePath);

  // Load ANN configuration with optional CLI overrides
  static ANN::CoreConfig<float> loadANNConfig(const std::string& configFilePath,
                                               std::optional<ANN::ModeType> modeType = std::nullopt,
                                               std::optional<ANN::DeviceType> deviceType = std::nullopt);

  // Load CNN configuration with optional CLI overrides
  static CNN::CoreConfig<float> loadCNNConfig(const std::string& configFilePath,
                                               std::optional<std::string> modeOverride = std::nullopt,
                                               std::optional<std::string> deviceOverride = std::nullopt);

  // Load ANN samples from JSON
  static ANN::Samples<float> loadANNSamples(const std::string& samplesFilePath);

  // Load CNN samples from JSON (requires inputShape for reshaping flat data to 3D)
  static CNN::Samples<float> loadCNNSamples(const std::string& samplesFilePath, const CNN::Shape3D& inputShape);

  // Load ANN input from JSON (flat vector)
  static ANN::Input<float> loadANNInput(const std::string& inputFilePath);

  // Load CNN input from JSON (flat vector reshaped to 3D tensor)
  static CNN::Input<float> loadCNNInput(const std::string& inputFilePath, const CNN::Shape3D& inputShape);
};

} // namespace NN_CLI

#endif // NN_CLI_LOADER_HPP

