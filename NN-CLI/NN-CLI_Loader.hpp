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
                                             const IOConfig& ioConfig,
                                             ulong progressReports = 1000);

  // Load CNN samples from JSON (supports image paths when ioConfig.inputType/outputType is IMAGE)
  static CNN::Samples<float> loadCNNSamples(const std::string& samplesFilePath,
                                             const CNN::Shape3D& inputShape,
                                             const IOConfig& ioConfig,
                                             ulong progressReports = 1000);

  // Load ANN inputs from JSON (batch: "inputs" array; supports image paths when ioConfig.inputType is IMAGE)
  static std::vector<ANN::Input<float>> loadANNInputs(const std::string& inputFilePath,
                                                       const IOConfig& ioConfig,
                                                       ulong progressReports = 1000);

  // Load CNN inputs from JSON (batch: "inputs" array; supports image paths when ioConfig.inputType is IMAGE)
  static std::vector<CNN::Input<float>> loadCNNInputs(const std::string& inputFilePath,
                                                       const CNN::Shape3D& inputShape,
                                                       const IOConfig& ioConfig,
                                                       ulong progressReports = 1000);

  // Load progressReports from config root (returns 1000 if not present)
  static ulong loadProgressReports(const std::string& configFilePath);

  // Load saveModelInterval from config root (returns 10 if not present; 0 = disabled)
  static ulong loadSaveModelInterval(const std::string& configFilePath);

  // Load data augmentation config from trainingConfig (NN-CLI handles augmentation, not ANN/CNN)
  struct AugmentationTransforms {
    bool  horizontalFlip = true;   // Mirror along vertical axis (true = enabled)
    float rotation       = 15.0f;  // Max rotation in degrees (0 = disabled, 15 = ±15°)
    float translation    = 0.1f;   // Max shift as fraction of image size (0 = disabled, 0.1 = ±10%)
    float brightness     = 0.1f;   // Max brightness delta (0 = disabled, 0.1 = ±0.1)
    float contrast       = 0.2f;   // Max contrast delta from 1.0 (0 = disabled, 0.2 = range 0.8–1.2×)
    float gaussianNoise  = 0.02f;  // Noise standard deviation (0 = disabled, 0.02 = σ=0.02)
  };

  struct AugmentationConfig {
    ulong augmentationFactor = 0;       // 0 = disabled; N = N× total samples per class
    bool balanceAugmentation = false;   // true = augment minority classes up to max class count
    bool autoClassWeights = false;      // true = auto-compute inverse-frequency class weights
    float augmentationProbability = 0.5f; // Probability of applying each enabled transform (default 50%)
    AugmentationTransforms transforms;  // Which transforms to apply and their intensities
  };
  static AugmentationConfig loadAugmentationConfig(const std::string& configFilePath);
};

} // namespace NN_CLI

#endif // NN_CLI_LOADER_HPP

