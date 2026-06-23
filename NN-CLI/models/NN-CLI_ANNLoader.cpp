#include "NN-CLI_ANNLoader.hpp"
#include "NN-CLI_Loader.hpp"
#include "NN-CLI_ImageLoader.hpp"
#include "NN-CLI_ModelSerializer.hpp"
#include "NN-CLI_TerminalUI_ProgressBar.hpp"

#include <QFile>
#include <QFileInfo>
#include <json.hpp>

#include <stdexcept>

namespace NN_CLI
{

  //===================================================================================================================//

  ANN::CoreConfig<float> ANNLoader::loadModel(const std::string& configFilePath,
                                              std::optional<Common::ModeType> modeType,
                                              std::optional<Common::DeviceType> deviceType)
  {
    return loadModel(Loader::parseConfigFile(configFilePath), modeType, deviceType);
  }

  //===================================================================================================================//

  ANN::CoreConfig<float> ANNLoader::loadModel(const nlohmann::json& json, std::optional<Common::ModeType> modeType,
                                              std::optional<Common::DeviceType> deviceType)
  {
    ANN::CoreConfig<float> coreConfig;

    if (json.contains("device")) {
      coreConfig.deviceType = Common::Device::nameToType(json.at("device").get<std::string>());
    } else {
      coreConfig.deviceType = Common::DeviceType::CPU;
    }

    if (json.contains("numThreads"))
      coreConfig.numThreads = json.at("numThreads").get<int>();

    if (json.contains("numGPUs"))
      coreConfig.numGPUs = json.at("numGPUs").get<int>();

    if (json.contains("mode")) {
      coreConfig.modeType = Common::Mode::nameToType(json.at("mode").get<std::string>());
    } else {
      coreConfig.modeType = Common::ModeType::PREDICT;
    }

    if (modeType.has_value())
      coreConfig.modeType = modeType.value();

    if (deviceType.has_value())
      coreConfig.deviceType = deviceType.value();

    if (!json.contains("layers")) {
      throw std::runtime_error("Config missing 'layers'");
    }

    for (const auto& layerJson : json.at("layers")) {
      ANN::Layer layer;
      layer.numNeurons = layerJson.at("numNeurons").get<ulong>();

      layer.actvFuncType = ANN::ActvFunc::nameToType(layerJson.at("actvFunc").get<std::string>());

      coreConfig.layersConfig.push_back(layer);
    }

    if (json.contains("costFunction")) {
      const auto& cfc = json.at("costFunction");
      coreConfig.costFunctionConfig.type = Common::CostFunction::nameToType(cfc.at("type").get<std::string>());

      if (cfc.contains("weights")) {
        coreConfig.costFunctionConfig.weights = cfc.at("weights").get<std::vector<float>>();
      }
    }

    coreConfig.trainConfig = Loader::loadTrainConfig(json);

    coreConfig.testConfig = Loader::loadTestConfig(json);

    coreConfig.calibrateConfig = Loader::loadCalibrateConfig(json);

    if (json.contains("parameters")) {
      throw std::runtime_error("This JSON file contains embedded parameters. "
                               "The embedded-parameter format is no longer supported. "
                               "Please use a .nnmodel.tar package with separate parameter files.");
    }

    return coreConfig;
  }

  //===================================================================================================================//

  ANN::CoreConfig<float> ANNLoader::loadModelPackage(const nlohmann::json& json, const std::vector<char>& binParams,
                                                     std::optional<Common::ModeType> modeType,
                                                     std::optional<Common::DeviceType> deviceType)
  {
    // 1. Call the existing JSON-only version to parse architecture/config
    auto coreConfig = loadModel(json, modeType, deviceType);

    // 2. If binary params provided, overwrite parameters from binary data
    if (!binParams.empty()) {
      ModelSerializer::loadANNParametersBinary(binParams, coreConfig, coreConfig.layersConfig);
    }

    // 3. Parse persisted training metadata and set startingEpoch for resume.
    coreConfig.loadedTrainMetadata = Loader::loadTrainMetadata(json);

    if (json.contains("trainMetadata") && json.at("trainMetadata").contains("lastEpoch"))
      coreConfig.trainConfig.startingEpoch = coreConfig.loadedTrainMetadata.lastEpoch + 1;

    return coreConfig;
  }

  //===================================================================================================================//

  ANN::Samples<float> ANNLoader::loadSamples(const std::string& samplesFilePath, const IOConfig& ioConfig,
                                             ulong progressReports)
  {
    QFile file(QString::fromStdString(samplesFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
      throw std::runtime_error("Failed to open samples file: " + samplesFilePath);
    }

    QByteArray fileData = file.readAll();
    nlohmann::json json = nlohmann::json::parse(fileData.toStdString());

    // Resolve base directory for relative image paths
    std::string baseDir = QFileInfo(QString::fromStdString(samplesFilePath)).absolutePath().toStdString();

    const auto& samplesArray = json.at("samples");
    size_t totalSamples = samplesArray.size();

    ANN::Samples<float> samples;
    samples.reserve(totalSamples);
    size_t idx = 0;

    for (const auto& sampleJson : samplesArray) {
      ANN::Sample<float> sample;

      // Input
      if (ioConfig.inputType == DataType::IMAGE) {
        if (!ioConfig.hasInputShape()) {
          throw std::runtime_error("inputType is 'image' but no inputShape provided in config.");
        }

        std::string imgPath = ImageLoader::resolvePath(sampleJson.at("input").get<std::string>(), baseDir);
        sample.input = ImageLoader::loadImage(imgPath, static_cast<int>(ioConfig.inputC),
                                              static_cast<int>(ioConfig.inputH), static_cast<int>(ioConfig.inputW));
      } else {
        sample.input = sampleJson.at("input").get<std::vector<float>>();
      }

      // Output
      if (ioConfig.outputType == DataType::IMAGE) {
        if (!ioConfig.hasOutputShape()) {
          throw std::runtime_error("outputType is 'image' but no outputShape provided in config.");
        }

        std::string imgPath = ImageLoader::resolvePath(sampleJson.at("output").get<std::string>(), baseDir);
        sample.output = ImageLoader::loadImage(imgPath, static_cast<int>(ioConfig.outputC),
                                               static_cast<int>(ioConfig.outputH), static_cast<int>(ioConfig.outputW));
      } else {
        sample.output = sampleJson.at("output").get<std::vector<float>>();
      }

      samples.push_back(std::move(sample));
      TerminalUI_ProgressBar::printLoadingProgress("Samples:", ++idx, totalSamples, progressReports);
    }

    return samples;
  }

  //===================================================================================================================//

  std::vector<ANN::Input<float>> ANNLoader::loadInputs(const std::string& inputFilePath, const IOConfig& ioConfig,
                                                       ulong progressReports)
  {
    QFile file(QString::fromStdString(inputFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
      throw std::runtime_error("Failed to open input file: " + inputFilePath);
    }

    QByteArray fileData = file.readAll();
    nlohmann::json json = nlohmann::json::parse(fileData.toStdString());

    const auto& inputsArray = json.at("inputs");

    if (!inputsArray.is_array() || inputsArray.empty()) {
      throw std::runtime_error("'inputs' must be a non-empty array in: " + inputFilePath);
    }

    std::string baseDir = QFileInfo(QString::fromStdString(inputFilePath)).absolutePath().toStdString();
    size_t totalInputs = inputsArray.size();
    std::vector<ANN::Input<float>> inputs;
    inputs.reserve(totalInputs);
    size_t idx = 0;

    for (const auto& entry : inputsArray) {
      if (ioConfig.inputType == DataType::IMAGE) {
        if (!ioConfig.hasInputShape()) {
          throw std::runtime_error("inputType is 'image' but no inputShape provided in config.");
        }

        std::string imgPath = ImageLoader::resolvePath(entry.get<std::string>(), baseDir);
        inputs.push_back(ImageLoader::loadImage(imgPath, static_cast<int>(ioConfig.inputC),
                                                static_cast<int>(ioConfig.inputH), static_cast<int>(ioConfig.inputW)));
      } else {
        inputs.push_back(entry.get<std::vector<float>>());
      }

      TerminalUI_ProgressBar::printLoadingProgress("Loading inputs:", ++idx, totalInputs, progressReports);
    }

    return inputs;
  }

  //===================================================================================================================//

} // namespace NN_CLI
