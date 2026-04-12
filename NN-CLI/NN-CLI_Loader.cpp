#include "NN-CLI_Loader.hpp"

#include <QFile>
#include <json.hpp>

#include <stdexcept>

namespace NN_CLI
{

  //===================================================================================================================//
  // Network type detection
  //===================================================================================================================//

  NetworkType Loader::detectNetworkType(const std::string& configFilePath)
  {
    QFile file(QString::fromStdString(configFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
      throw std::runtime_error("Failed to open config file: " + configFilePath);
    }

    QByteArray fileData = file.readAll();
    nlohmann::json json = nlohmann::json::parse(fileData.toStdString());

    // ANN configs use "layersConfig" for their dense layers.
    // CNN configs use "convolutionalLayersConfig" and/or "denseLayersConfig".
    // "inputShape" is NOT used for detection — both types can have it (e.g. ANN image input).
    if (json.contains("layersConfig")) {
      return NetworkType::ANN;
    }

    return NetworkType::CNN;
  }

  //===================================================================================================================//
  // I/O config loading
  //===================================================================================================================//

  IOConfig Loader::loadIOConfig(const std::string& configFilePath, std::optional<std::string> inputTypeOverride,
                                std::optional<std::string> outputTypeOverride)
  {
    QFile file(QString::fromStdString(configFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
      throw std::runtime_error("Failed to open config file: " + configFilePath);
    }

    QByteArray fileData = file.readAll();
    nlohmann::json json = nlohmann::json::parse(fileData.toStdString());

    IOConfig ioConfig;

    // Read inputType / outputType (default to "vector")
    if (json.contains("inputType")) {
      ioConfig.inputType = dataTypeFromString(json.at("inputType").get<std::string>());
    }

    if (json.contains("outputType")) {
      ioConfig.outputType = dataTypeFromString(json.at("outputType").get<std::string>());
    }

    // CLI overrides
    if (inputTypeOverride.has_value()) {
      ioConfig.inputType = dataTypeFromString(inputTypeOverride.value());
    }

    if (outputTypeOverride.has_value()) {
      ioConfig.outputType = dataTypeFromString(outputTypeOverride.value());
    }

    // Input shape (for ANN image input — CNN uses CoreConfig.inputShape)
    if (json.contains("inputShape")) {
      const auto& s = json.at("inputShape");
      ioConfig.inputC = s.at("c").get<ulong>();
      ioConfig.inputH = s.at("h").get<ulong>();
      ioConfig.inputW = s.at("w").get<ulong>();
    }

    // Output shape (for image output reconstruction)
    if (json.contains("outputShape")) {
      const auto& s = json.at("outputShape");
      ioConfig.outputC = s.at("c").get<ulong>();
      ioConfig.outputH = s.at("h").get<ulong>();
      ioConfig.outputW = s.at("w").get<ulong>();
    }

    return ioConfig;
  }

  //===================================================================================================================//
  // progressReports loading
  //===================================================================================================================//

  ulong Loader::loadProgressReports(const std::string& configFilePath)
  {
    QFile file(QString::fromStdString(configFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
      throw std::runtime_error("Failed to open config file: " + configFilePath);
    }

    QByteArray fileData = file.readAll();
    nlohmann::json json = nlohmann::json::parse(fileData.toStdString());

    if (json.contains("progressReports")) {
      return json.at("progressReports").get<ulong>();
    }

    return 1000; // default
  }

  //===================================================================================================================//
  // saveModelInterval loading
  //===================================================================================================================//

  ulong Loader::loadSaveModelInterval(const std::string& configFilePath)
  {
    QFile file(QString::fromStdString(configFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
      throw std::runtime_error("Failed to open config file: " + configFilePath);
    }

    QByteArray fileData = file.readAll();
    nlohmann::json json = nlohmann::json::parse(fileData.toStdString());

    if (json.contains("saveModelInterval")) {
      return json.at("saveModelInterval").get<ulong>();
    }

    return 10; // default: save every 10 epochs
  }

  //===================================================================================================================//
  // Augmentation config loading
  //===================================================================================================================//

  AugmentationConfig Loader::loadAugmentationConfig(const std::string& configFilePath)
  {
    QFile file(QString::fromStdString(configFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
      throw std::runtime_error("Failed to open config file: " + configFilePath);
    }

    QByteArray fileData = file.readAll();
    nlohmann::json json = nlohmann::json::parse(fileData.toStdString());

    AugmentationConfig config;

    if (json.contains("trainingConfig")) {
      const auto& tc = json.at("trainingConfig");

      if (tc.contains("augmentationFactor"))
        config.augmentationFactor = tc.at("augmentationFactor").get<ulong>();

      if (tc.contains("balanceAugmentation"))
        config.balanceAugmentation = tc.at("balanceAugmentation").get<bool>();

      if (tc.contains("autoClassWeights"))
        config.autoClassWeights = tc.at("autoClassWeights").get<bool>();

      if (tc.contains("augmentationProbability"))
        config.augmentationProbability = tc.at("augmentationProbability").get<float>();

      if (tc.contains("augmentationTransforms")) {
        const auto& at = tc.at("augmentationTransforms");
        auto& t = config.transforms;

        if (at.contains("horizontalFlip"))
          t.horizontalFlip = at.at("horizontalFlip").get<bool>();

        if (at.contains("rotation"))
          t.rotation = at.at("rotation").get<float>();

        if (at.contains("translation"))
          t.translation = at.at("translation").get<float>();

        if (at.contains("brightness"))
          t.brightness = at.at("brightness").get<float>();

        if (at.contains("contrast"))
          t.contrast = at.at("contrast").get<float>();

        if (at.contains("gaussianNoise"))
          t.gaussianNoise = at.at("gaussianNoise").get<float>();
      }

      if (tc.contains("validationDataset")) {
        const auto& vd = tc.at("validationDataset");
        auto& vc = config.validationDataset;

        if (vd.contains("enabled"))
          vc.enabled = vd.at("enabled").get<bool>();

        if (vd.contains("autoSize"))
          vc.autoSize = vd.at("autoSize").get<bool>();

        if (vd.contains("size"))
          vc.size = vd.at("size").get<float>();

        if (vd.contains("checkInterval"))
          vc.checkInterval = vd.at("checkInterval").get<ulong>();
      }
    }

    return config;
  }

  //===================================================================================================================//

} // namespace NN_CLI
