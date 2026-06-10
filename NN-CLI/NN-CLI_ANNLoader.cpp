#include "NN-CLI_ANNLoader.hpp"
#include "NN-CLI_Loader.hpp"
#include "NN-CLI_ImageLoader.hpp"
#include "NN-CLI_ModelSerializer.hpp"
#include "NN-CLI_ProgressBar.hpp"

#include <QFile>
#include <QFileInfo>
#include <json.hpp>

#include <stdexcept>

namespace NN_CLI
{

  //===================================================================================================================//

  ANN::CoreConfig<float> ANNLoader::loadConfig(const std::string& configFilePath,
                                               std::optional<Common::ModeType> modeType,
                                               std::optional<Common::DeviceType> deviceType)
  {
    return loadConfig(Loader::parseConfigFile(configFilePath), modeType, deviceType);
  }

  //===================================================================================================================//

  ANN::CoreConfig<float> ANNLoader::loadConfig(const nlohmann::json& json, std::optional<Common::ModeType> modeType,
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

    if (json.contains("training")) {
      const auto& tc = json.at("training");
      coreConfig.trainingConfig.numEpochs = tc.at("numEpochs").get<ulong>();
      coreConfig.trainingConfig.learningRate = tc.at("learningRate").get<float>();

      if (tc.contains("batchSize"))
        coreConfig.trainingConfig.batchSize = tc.at("batchSize").get<ulong>();

      if (tc.contains("shuffleSamples"))
        coreConfig.trainingConfig.shuffleSamples = tc.at("shuffleSamples").get<bool>();

      if (tc.contains("shuffleSeed"))
        coreConfig.trainingConfig.shuffleSeed = tc.at("shuffleSeed").get<uint32_t>();

      if (tc.contains("dropoutRate"))
        coreConfig.trainingConfig.dropoutRate = tc.at("dropoutRate").get<float>();

      if (tc.contains("optimizer")) {
        const auto& opt = tc.at("optimizer");

        if (opt.contains("type"))
          coreConfig.trainingConfig.optimizer.type =
            Common::Optimizer<float>::nameToType(opt.at("type").get<std::string>());

        if (opt.contains("beta1"))
          coreConfig.trainingConfig.optimizer.beta1 = opt.at("beta1").get<float>();

        if (opt.contains("beta2"))
          coreConfig.trainingConfig.optimizer.beta2 = opt.at("beta2").get<float>();

        if (opt.contains("epsilon"))
          coreConfig.trainingConfig.optimizer.epsilon = opt.at("epsilon").get<float>();
      }

      if (tc.contains("monitoring")) {
        const auto& mon = tc.at("monitoring");
        auto& mc = coreConfig.trainingConfig.monitoringConfig;

        if (mon.contains("enabled"))
          mc.enabled = mon.at("enabled").get<bool>();

        if (mon.contains("checkInterval"))
          mc.checkInterval = mon.at("checkInterval").get<ulong>();

        if (mon.contains("patience"))
          mc.patience = mon.at("patience").get<ulong>();

        if (mon.contains("metrics")) {
          const auto& metrics = mon.at("metrics");

          if (metrics.contains("lossStagnation")) {
            const auto& ls = metrics.at("lossStagnation");

            if (ls.contains("enabled"))
              mc.metrics.lossStagnation.enabled = ls.at("enabled").get<bool>();

            if (ls.contains("minDelta"))
              mc.metrics.lossStagnation.minDelta = ls.at("minDelta").get<float>();
          }

          if (metrics.contains("lossExplosion")) {
            const auto& le = metrics.at("lossExplosion");

            if (le.contains("enabled"))
              mc.metrics.lossExplosion.enabled = le.at("enabled").get<bool>();

            if (le.contains("threshold"))
              mc.metrics.lossExplosion.threshold = le.at("threshold").get<float>();
          }
        }
      }
    }

    if (json.contains("test")) {
      const auto& tc = json.at("test");

      if (tc.contains("batchSize"))
        coreConfig.testConfig.batchSize = tc.at("batchSize").get<ulong>();
    }

    if (json.contains("parameters")) {
      throw std::runtime_error("This JSON file contains embedded parameters. "
                               "The embedded-parameter format is no longer supported. "
                               "Please use a .nnmodel.tar package with separate parameter files.");
    }

    return coreConfig;
  }

  //===================================================================================================================//

  ANN::CoreConfig<float> ANNLoader::loadConfig(const nlohmann::json& json, const std::vector<char>& binParams,
                                               std::optional<Common::ModeType> modeType,
                                               std::optional<Common::DeviceType> deviceType)
  {
    // 1. Call the existing JSON-only version to parse architecture/config
    auto coreConfig = loadConfig(json, modeType, deviceType);

    // 2. If binary params provided, overwrite parameters from binary data
    if (!binParams.empty()) {
      ModelSerializer::loadANNParametersBinary(binParams, coreConfig, coreConfig.layersConfig);
    }

    // 3. Parse epoch history and set startingEpoch from saved training metadata
    if (json.contains("trainingMetadata")) {
      const auto& md = json.at("trainingMetadata");

      // Set startingEpoch from lastEpoch: lastEpoch is the 0-based index of the
      // last completed epoch, so resume on the next one. E.g. lastEpoch=24
      // (epochs 0..24 done) resumes at startingEpoch=25 (e=25..99 = 75 more).
      if (md.contains("lastEpoch")) {
        ulong lastEpoch = md.at("lastEpoch").get<ulong>();
        coreConfig.trainingConfig.startingEpoch = lastEpoch + 1;
      }

      // Parse epoch history array
      if (md.contains("epochs") && md.at("epochs").is_array()) {
        const auto& epochsArr = md.at("epochs");
        coreConfig.loadedEpochHistory.reserve(epochsArr.size());

        for (const auto& recordJson : epochsArr) {
          Common::EpochRecord<float> record;
          record.epoch = recordJson.at("epoch").get<ulong>();
          record.loss = recordJson.at("loss").get<float>();

          if (recordJson.contains("valLoss") && recordJson.value("hasValLoss", false)) {
            record.valLoss = recordJson.at("valLoss").get<float>();
            record.hasValLoss = true;
          }

          record.isBest = recordJson.value("isBest", false);
          record.completionTime = recordJson.value("completionTime", 0UL);

          coreConfig.loadedEpochHistory.push_back(record);
        }
      }
    }

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
      ProgressBar::printLoadingProgress("Samples:", ++idx, totalSamples, progressReports);
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

      ProgressBar::printLoadingProgress("Loading inputs:", ++idx, totalInputs, progressReports);
    }

    return inputs;
  }

  //===================================================================================================================//

} // namespace NN_CLI
