#ifndef ANN_CLI_LOADER_HPP
#define ANN_CLI_LOADER_HPP

#include <ANN_Core.hpp>
#include <ANN_CoreMode.hpp>
#include <ANN_CoreType.hpp>
#include <ANN_ActvFunc.hpp>
#include <ANN_LayersConfig.hpp>

#include <string>

namespace ANN_CLI {

class Loader {
public:
  // Load config file for training (architecture + training settings, no weights)
  static ANN::CoreConfig<float> loadConfig(const std::string& configFilePath,
                                           ANN::CoreModeType modeType,
                                           ANN::CoreTypeType coreType);

  // Load trained model file for run/test modes (architecture + trained weights)
  static ANN::CoreConfig<float> loadModel(const std::string& modelFilePath,
                                          ANN::CoreTypeType coreType);

  static ANN::Samples<float> loadSamples(const std::string& samplesFilePath);

  static ANN::Input<float> loadInput(const std::string& inputFilePath);
};

} // namespace ANN_CLI

#endif // ANN_CLI_LOADER_HPP

