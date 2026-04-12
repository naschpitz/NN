#include "NN-CLI_Runner.hpp"

#include "NN-CLI_ANNLoader.hpp"
#include "NN-CLI_ANNRunner.hpp"
#include "NN-CLI_CNNLoader.hpp"
#include "NN-CLI_CNNRunner.hpp"
#include "NN-CLI_Loader.hpp"

#include <iostream>
#include <optional>
#include <string>

using namespace NN_CLI;

//===================================================================================================================//

Runner::Runner(const QCommandLineParser& parser, LogLevel logLevel) : parser(parser), logLevel(logLevel)
{
  QString configPath = this->parser.value("config");

  // Detect network type from config file
  this->networkType = Loader::detectNetworkType(configPath.toStdString());

  // Build optional mode/device overrides as strings
  std::optional<std::string> modeOverride;

  if (this->parser.isSet("mode")) {
    modeOverride = this->parser.value("mode").toLower().toStdString();
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

  this->ioConfig = Loader::loadIOConfig(configPath.toStdString(), inputTypeOverride, outputTypeOverride);

  // Display info (verbose level >= 1)
  std::string networkTypeStr = (this->networkType == NetworkType::CNN) ? "CNN" : "ANN";
  std::string modeDisplay = modeOverride.has_value() ? (modeOverride.value() + " (CLI)") : "from config file";
  std::string deviceDisplay = deviceOverride.has_value() ? (deviceOverride.value() + " (CLI)") : "from config file";

  if (this->logLevel >= LogLevel::INFO) {
    std::cout << "Network type: " << networkTypeStr << "\n";
    std::cout << "Loading configuration from: " << configPath.toStdString() << "\n";
    std::cout << "Mode: " << modeDisplay << ", Device: " << deviceDisplay << "\n";
    std::cout << "Input type: " << dataTypeToString(this->ioConfig.inputType)
              << ", Output type: " << dataTypeToString(this->ioConfig.outputType) << "\n";
  }

  // Load NN-CLI-level settings from config root
  this->ioConfig.progressReports = Loader::loadProgressReports(configPath.toStdString());
  this->ioConfig.saveModelInterval = Loader::loadSaveModelInterval(configPath.toStdString());

  // Load data augmentation config
  this->augConfig = Loader::loadAugmentationConfig(configPath.toStdString());

  if (this->logLevel >= LogLevel::INFO && this->ioConfig.saveModelInterval > 0) {
    std::cout << "Save model interval: every " << this->ioConfig.saveModelInterval << " epoch(s)\n";
  }

  if (this->networkType == NetworkType::ANN) {
    // Convert string overrides to ANN enum overrides
    std::optional<ANN::ModeType> annModeOverride;

    if (modeOverride.has_value())
      annModeOverride = ANN::Mode::nameToType(modeOverride.value());

    std::optional<ANN::DeviceType> annDeviceOverride;

    if (deviceOverride.has_value())
      annDeviceOverride = ANN::Device::nameToType(deviceOverride.value());

    this->annCoreConfig = ANNLoader::loadConfig(configPath.toStdString(), annModeOverride, annDeviceOverride);
    this->annCoreConfig.logLevel = static_cast<ANN::LogLevel>(this->logLevel);
    this->mode = ANN::Mode::typeToName(this->annCoreConfig.modeType);
    this->annCore = ANN::Core<float>::makeCore(this->annCoreConfig);
  } else {
    this->cnnCoreConfig = CNNLoader::loadConfig(configPath.toStdString(), modeOverride, deviceOverride);
    this->cnnCoreConfig.logLevel = static_cast<CNN::LogLevel>(this->logLevel);
    this->mode = CNN::Mode::typeToName(this->cnnCoreConfig.modeType);
    this->cnnCore = CNN::Core<float>::makeCore(this->cnnCoreConfig);
  }
}

//===================================================================================================================//

int Runner::run()
{
  if (this->networkType == NetworkType::ANN) {
    ANNRunner annRunner(this->parser, this->logLevel, this->ioConfig, this->augConfig, this->annCore,
                        this->annCoreConfig);

    if (this->mode == "train")
      return annRunner.train();

    if (this->mode == "test")
      return annRunner.test();

    return annRunner.predict();
  } else {
    CNNRunner cnnRunner(this->parser, this->logLevel, this->ioConfig, this->augConfig, this->cnnCore,
                        this->cnnCoreConfig);

    if (this->mode == "train")
      return cnnRunner.train();

    if (this->mode == "test")
      return cnnRunner.test();

    return cnnRunner.predict();
  }
}
