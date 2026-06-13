#include "NN-CLI_App.hpp"

#include "NN-CLI_ANNLoader.hpp"
#include "NN-CLI_ANNRunner.hpp"
#include "NN-CLI_CalibrateController.hpp"
#include "NN-CLI_CalibrateRunner.hpp"
#include "NN-CLI_CNNLoader.hpp"
#include "NN-CLI_CNNRunner.hpp"
#include "NN-CLI_Loader.hpp"
#include "NN-CLI_ModelPackage.hpp"
#include "NN-CLI_ModelSerializer.hpp"
#include "NN-CLI_PredictController.hpp"
#include "NN-CLI_TestController.hpp"
#include "NN-CLI_TrainingController.hpp"

#include <filesystem>
#include <iostream>
#include <optional>
#include <string>

namespace fs = std::filesystem;

using namespace NN_CLI;

//===================================================================================================================//

App::App(const QCommandLineParser& parser, LogLevel logLevel) : parser(parser), logLevel(logLevel)
{
  QString configPath = this->parser.value("config");
  std::string configStr = configPath.toStdString();

  // Detect if input is a .nnmodel package
  bool isPackage = ModelPackage::isPackage(configStr);

  nlohmann::json json;
  std::vector<char> packageBinData;

  if (isPackage) {
    // Extract JSON config from package
    std::string jsonStr = ModelPackage::readJsonFromPackage(configStr);
    json = nlohmann::json::parse(jsonStr);

    // Extract binary parameters from package
    packageBinData = ModelPackage::readBinaryFromPackage(configStr);
  } else {
    // Plain JSON file
    json = Loader::parseConfigFile(configStr);
  }

  // Detect network type from config
  this->networkType = Loader::detectNetworkType(json);

  // Build optional mode/device overrides as strings
  std::optional<std::string> modeOverride;
  std::string modeDisplay = "from config file";

  if (this->parser.isSet("mode")) {
    modeOverride = this->parser.value("mode").toLower().toStdString();
    modeDisplay = this->parser.value("mode").toLower().toStdString() + " (CLI)";
  }

  // "calibrate" is a CLI-only mode that internally drives predict. The
  // model's underlying ModeType enum (TRAIN/TEST/PREDICT) doesn't know
  // about it, so we redirect the override to "predict" before passing
  // it down and remember locally that we're in calibrate mode.
  if (modeOverride.has_value() && modeOverride.value() == "calibrate") {
    this->isCalibrateMode = true;
    modeOverride = std::string("predict");
  }

  std::optional<std::string> deviceOverride;

  if (this->parser.isSet("device")) {
    deviceOverride = this->parser.value("device").toLower().toStdString();
  }

  // Load I/O config (inputType, outputType, shapes) with optional CLI overrides
  std::optional<std::string> inputTypeOverride;

  if (this->parser.isSet("input-type")) {
    inputTypeOverride = this->parser.value("input-type").toLower().toStdString();
  }

  std::optional<std::string> outputTypeOverride;

  if (this->parser.isSet("output-type")) {
    outputTypeOverride = this->parser.value("output-type").toLower().toStdString();
  }

  this->ioConfig = Loader::loadIOConfig(json, inputTypeOverride, outputTypeOverride);

  // Display info (verbose level >= 1)
  std::string networkTypeStr = (this->networkType == NetworkType::CNN) ? "CNN" : "ANN";
  std::string deviceDisplay = deviceOverride.has_value() ? (deviceOverride.value() + " (CLI)") : "from config file";

  if (this->logLevel >= LogLevel::INFO) {
    std::cout << "Network type: " << networkTypeStr << "\n";
    std::cout << "Loading configuration from: " << configStr << "\n";
    std::cout << "Mode: " << modeDisplay << ", Device: " << deviceDisplay << "\n";
    std::cout << "Input type: " << dataTypeToString(this->ioConfig.inputType)
              << ", Output type: " << dataTypeToString(this->ioConfig.outputType) << "\n";
  }

  // Load NN-CLI-level settings from config root
  this->ioConfig.progressReports = Loader::loadProgressReports(json);
  this->ioConfig.saveModelInterval = Loader::loadSaveModelInterval(json);

  // Load data augmentation config
  this->augConfig = Loader::loadAugmentationConfig(json);

  if (this->logLevel >= LogLevel::INFO && this->ioConfig.saveModelInterval > 0) {
    std::cout << "Save model interval: every " << this->ioConfig.saveModelInterval << " epoch(s)\n";
  }

  if (this->networkType == NetworkType::ANN) {
    // Convert string overrides to enum overrides
    std::optional<Common::ModeType> annModeOverride;

    if (modeOverride.has_value())
      annModeOverride = Common::Mode::nameToType(modeOverride.value());

    std::optional<Common::DeviceType> annDeviceOverride;

    if (deviceOverride.has_value())
      annDeviceOverride = Common::Device::nameToType(deviceOverride.value());

    if (isPackage) {
      this->annCoreConfig = ANNLoader::loadConfig(json, packageBinData, annModeOverride, annDeviceOverride);
    } else {
      this->annCoreConfig = ANNLoader::loadConfig(json, annModeOverride, annDeviceOverride);
    }

    // Validate parameters for predict/test in non-package mode (plain JSON).
    // Packages store params in binary, but a plain JSON without parameters would
    // produce zero-initialized weights -> garbage predictions.
    if (!isPackage && (this->annCoreConfig.modeType == Common::ModeType::PREDICT ||
                       this->annCoreConfig.modeType == Common::ModeType::TEST)) {
      if (this->annCoreConfig.parameters.weights.empty() || this->annCoreConfig.parameters.biases.empty()) {
        throw std::runtime_error("Config missing parameters required for predict/test mode. "
                                 "Use a .nnmodel package for predict/test mode. Plain JSON "
                                 "without parameters cannot be used for inference.");
      }
    }

    this->annCoreConfig.logLevel = static_cast<Common::LogLevel>(this->logLevel);
    this->mode = Common::Mode::typeToName(this->annCoreConfig.modeType);
    this->annCore = ANN::Core<float>::makeCore(this->annCoreConfig);
  } else {
    if (isPackage) {
      this->cnnCoreConfig = CNNLoader::loadConfig(json, packageBinData, modeOverride, deviceOverride);
    } else {
      this->cnnCoreConfig = CNNLoader::loadConfig(json, modeOverride, deviceOverride);
    }

    // Validate parameters for predict/test in non-package mode (plain JSON).
    // Packages store params in binary, but a plain JSON without parameters would
    // produce zero-initialized weights -> garbage predictions.
    if (!isPackage && (this->cnnCoreConfig.modeType == Common::ModeType::PREDICT ||
                       this->cnnCoreConfig.modeType == Common::ModeType::TEST)) {
      bool hasParams = !this->cnnCoreConfig.parameters.convParams.empty() ||
                       !this->cnnCoreConfig.parameters.denseParams.weights.empty();
      if (!hasParams) {
        throw std::runtime_error("CNN config missing parameters required for predict/test mode. "
                                 "Use a .nnmodel package or provide 'parameters' in the JSON config.");
      }
    }

    this->cnnCoreConfig.logLevel = static_cast<Common::LogLevel>(this->logLevel);
    this->mode = Common::Mode::typeToName(this->cnnCoreConfig.modeType);
    this->cnnCore = CNN::Core<float>::makeCore(this->cnnCoreConfig);
  }
}

//===================================================================================================================//

int App::run()
{
  if (this->networkType == NetworkType::ANN) {
    auto runner = std::make_unique<ANNRunner>(this->parser, this->logLevel, this->ioConfig, this->augConfig,
                                              this->annCore, this->annCoreConfig);

    if (this->isCalibrateMode) {
      // Merge CLI overrides into coreConfig.calibrationConfig
      this->annCoreConfig.calibrationConfig.idImagesDir = this->parser.value("id-images").toStdString();
      this->annCoreConfig.calibrationConfig.oodDir = this->parser.isSet("ood-dir")
                                                        ? this->parser.value("ood-dir").toStdString()
                                                        : (fs::current_path() / "extern-datasets" / "ood").string();
      this->annCoreConfig.calibrationConfig.idSampleCount =
          this->parser.isSet("id-sample-count") ? this->parser.value("id-sample-count").toULongLong() : 500;
      this->annCoreConfig.calibrationConfig.oodSampleCount =
          this->parser.isSet("ood-sample-count") ? this->parser.value("ood-sample-count").toULongLong() : 1500;
      this->annCoreConfig.calibrationConfig.idPercentile =
          this->parser.isSet("id-percentile") ? this->parser.value("id-percentile").toDouble() : 95.0;
      this->annCoreConfig.calibrationConfig.fetchIfMissing = !this->parser.isSet("no-fetch");
      this->annCoreConfig.calibrationConfig.outputPath = this->parser.isSet("output")
                                                            ? this->parser.value("output").toStdString()
                                                            : (fs::path(this->parser.value("config").toStdString()).parent_path() /
                                                               "threshold.json").string();
      this->annCoreConfig.calibrationConfig.logLevel = static_cast<Common::LogLevel>(this->logLevel);
      this->annCoreConfig.calibrationConfig.progressReports = this->ioConfig.progressReports;

      CalibrateController<ANNRunner> ctrl;
      ctrl.init(std::move(runner));
      return ctrl.startCalibrate();
    }

    if (this->mode == "train") {
      TrainingController<ANNRunner> controller;
      controller.init(std::move(runner));
      return controller.startTraining();
    }

    if (this->mode == "test") {
      TestController<ANNRunner> controller;
      controller.init(std::move(runner));
      return controller.startTest();
    }

    PredictController<ANNRunner> controller;
    controller.init(std::move(runner));
    return controller.startPredict();
  } else {
    auto runner = std::make_unique<CNNRunner>(this->parser, this->logLevel, this->ioConfig, this->augConfig,
                                              this->cnnCore, this->cnnCoreConfig);

    if (this->isCalibrateMode) {
      // Merge CLI overrides into coreConfig.calibrationConfig
      this->cnnCoreConfig.calibrationConfig.idImagesDir = this->parser.value("id-images").toStdString();
      this->cnnCoreConfig.calibrationConfig.oodDir = this->parser.isSet("ood-dir")
                                                       ? this->parser.value("ood-dir").toStdString()
                                                       : (fs::current_path() / "extern-datasets" / "ood").string();
      this->cnnCoreConfig.calibrationConfig.idSampleCount =
          this->parser.isSet("id-sample-count") ? this->parser.value("id-sample-count").toULongLong() : 500;
      this->cnnCoreConfig.calibrationConfig.oodSampleCount =
          this->parser.isSet("ood-sample-count") ? this->parser.value("ood-sample-count").toULongLong() : 1500;
      this->cnnCoreConfig.calibrationConfig.idPercentile =
          this->parser.isSet("id-percentile") ? this->parser.value("id-percentile").toDouble() : 95.0;
      this->cnnCoreConfig.calibrationConfig.fetchIfMissing = !this->parser.isSet("no-fetch");
      this->cnnCoreConfig.calibrationConfig.outputPath = this->parser.isSet("output")
                                                            ? this->parser.value("output").toStdString()
                                                            : (fs::path(this->parser.value("config").toStdString()).parent_path() /
                                                               "threshold.json").string();
      this->cnnCoreConfig.calibrationConfig.logLevel = static_cast<Common::LogLevel>(this->logLevel);
      this->cnnCoreConfig.calibrationConfig.progressReports = this->ioConfig.progressReports;

      CalibrateController<CNNRunner> ctrl;
      ctrl.init(std::move(runner));
      return ctrl.startCalibrate();
    }

    if (this->mode == "train") {
      TrainingController<CNNRunner> controller;
      controller.init(std::move(runner));
      return controller.startTraining();
    }

    if (this->mode == "test") {
      TestController<CNNRunner> controller;
      controller.init(std::move(runner));
      return controller.startTest();
    }

    PredictController<CNNRunner> controller;
    controller.init(std::move(runner));
    return controller.startPredict();
  }
}
