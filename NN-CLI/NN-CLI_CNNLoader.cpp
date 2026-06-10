#include "NN-CLI_CNNLoader.hpp"
#include "NN-CLI_ImageLoader.hpp"
#include "NN-CLI_Loader.hpp"
#include "NN-CLI_ModelSerializer.hpp"
#include "NN-CLI_ProgressBar.hpp"

#include <QFile>
#include <QFileInfo>
#include <json.hpp>

#include <stdexcept>

namespace NN_CLI
{

  //===================================================================================================================//
  // CNN config loading
  //===================================================================================================================//

  CNN::CoreConfig<float> CNNLoader::loadConfig(const std::string& configFilePath,
                                               std::optional<std::string> modeOverride,
                                               std::optional<std::string> deviceOverride)
  {
    return loadConfig(Loader::parseConfigFile(configFilePath), modeOverride, deviceOverride);
  }

  //===================================================================================================================//

  CNN::CoreConfig<float> CNNLoader::loadConfig(const nlohmann::json& json, std::optional<std::string> modeOverride,
                                               std::optional<std::string> deviceOverride)
  {
    CNN::CoreConfig<float> coreConfig;

    // Device
    if (deviceOverride.has_value()) {
      coreConfig.deviceType = Common::Device::nameToType(deviceOverride.value());
    } else if (json.contains("device")) {
      coreConfig.deviceType = Common::Device::nameToType(json.at("device").get<std::string>());
    } else {
      coreConfig.deviceType = Common::DeviceType::CPU;
    }

    if (json.contains("numThreads"))
      coreConfig.numThreads = json.at("numThreads").get<int>();

    if (json.contains("numGPUs"))
      coreConfig.numGPUs = json.at("numGPUs").get<int>();

    // Mode
    if (modeOverride.has_value()) {
      coreConfig.modeType = Common::Mode::nameToType(modeOverride.value());
    } else if (json.contains("mode")) {
      coreConfig.modeType = Common::Mode::nameToType(json.at("mode").get<std::string>());
    } else {
      coreConfig.modeType = Common::ModeType::PREDICT;
    }

    // Input shape (required for CNN)
    if (!json.contains("inputShape")) {
      throw std::runtime_error("CNN config missing 'inputShape'");
    }

    const auto& shapeJson = json.at("inputShape");
    coreConfig.inputShape.c = shapeJson.at("c").get<ulong>();
    coreConfig.inputShape.h = shapeJson.at("h").get<ulong>();
    coreConfig.inputShape.w = shapeJson.at("w").get<ulong>();

    // CNN layers
    if (json.contains("convolutionalLayers")) {
      for (const auto& layerJson : json.at("convolutionalLayers")) {
        std::string type = layerJson.at("type").get<std::string>();
        CNN::CNNLayerConfig layerConfig;

        if (type == "conv") {
          layerConfig.type = CNN::LayerType::CONV;
          CNN::ConvLayerConfig conv;
          conv.numFilters = layerJson.at("numFilters").get<ulong>();
          conv.filterH = layerJson.at("filterH").get<ulong>();
          conv.filterW = layerJson.at("filterW").get<ulong>();
          conv.strideY = layerJson.at("strideY").get<ulong>();
          conv.strideX = layerJson.at("strideX").get<ulong>();
          conv.slidingStrategy = CNN::SlidingStrategy::nameToType(layerJson.at("slidingStrategy").get<std::string>());
          layerConfig.config = conv;
        } else if (type == "relu") {
          layerConfig.type = CNN::LayerType::RELU;
          layerConfig.config = CNN::ReLULayerConfig{};
        } else if (type == "pool") {
          layerConfig.type = CNN::LayerType::POOL;
          CNN::PoolLayerConfig pool;
          pool.poolType = CNN::PoolType::nameToType(layerJson.at("poolType").get<std::string>());
          pool.poolH = layerJson.at("poolH").get<ulong>();
          pool.poolW = layerJson.at("poolW").get<ulong>();
          pool.strideY = layerJson.at("strideY").get<ulong>();
          pool.strideX = layerJson.at("strideX").get<ulong>();
          layerConfig.config = pool;
        } else if (type == "flatten") {
          layerConfig.type = CNN::LayerType::FLATTEN;
          layerConfig.config = CNN::FlattenLayerConfig{};
        } else if (type == "globalavgpool") {
          layerConfig.type = CNN::LayerType::GLOBALAVGPOOL;
          layerConfig.config = CNN::GlobalAvgPoolLayerConfig{};
        } else if (type == "globaldualpool") {
          layerConfig.type = CNN::LayerType::GLOBALDUALPOOL;
          layerConfig.config = CNN::GlobalDualPoolLayerConfig{};
        } else if (type == "instancenorm") {
          layerConfig.type = CNN::LayerType::INSTANCENORM;
          CNN::NormLayerConfig bn;

          if (layerJson.contains("epsilon"))
            bn.epsilon = layerJson.at("epsilon").get<float>();

          if (layerJson.contains("momentum"))
            bn.momentum = layerJson.at("momentum").get<float>();

          layerConfig.config = bn;
        } else if (type == "batchnorm") {
          layerConfig.type = CNN::LayerType::BATCHNORM;
          CNN::NormLayerConfig bn;

          if (layerJson.contains("epsilon"))
            bn.epsilon = layerJson.at("epsilon").get<float>();

          if (layerJson.contains("momentum"))
            bn.momentum = layerJson.at("momentum").get<float>();

          layerConfig.config = bn;
        } else if (type == "residual_start") {
          layerConfig.type = CNN::LayerType::RESIDUAL_START;
          layerConfig.config = CNN::ResidualStartConfig{};
        } else if (type == "residual_end") {
          layerConfig.type = CNN::LayerType::RESIDUAL_END;
          layerConfig.config = CNN::ResidualEndConfig{};
        } else {
          throw std::runtime_error("Unknown CNN layer type: " + type);
        }

        coreConfig.layersConfig.cnnLayers.push_back(layerConfig);
      }
    }

    // Dense layers
    if (json.contains("denseLayers")) {
      for (const auto& layerJson : json.at("denseLayers")) {
        CNN::DenseLayerConfig dense;
        dense.numNeurons = layerJson.at("numNeurons").get<ulong>();

        dense.actvFuncType = ANN::ActvFunc::nameToType(layerJson.at("actvFunc").get<std::string>());

        coreConfig.layersConfig.denseLayers.push_back(dense);
      }
    }

    // Cost function config
    if (json.contains("costFunction")) {
      const auto& cfc = json.at("costFunction");
      coreConfig.costFunctionConfig.type = Common::CostFunction::nameToType(cfc.at("type").get<std::string>());

      if (cfc.contains("weights")) {
        coreConfig.costFunctionConfig.weights = cfc.at("weights").get<std::vector<float>>();
      }
    }

    // Training config
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
                               "Please use a .nnmodel package with separate parameter files.");
    }

    return coreConfig;
  }

  //===================================================================================================================//

  CNN::CoreConfig<float> CNNLoader::loadConfig(const nlohmann::json& json, const std::vector<char>& binParams,
                                               std::optional<std::string> modeOverride,
                                               std::optional<std::string> deviceOverride)
  {
    // 1. Call the existing JSON-only version to parse architecture/config
    auto coreConfig = loadConfig(json, modeOverride, deviceOverride);

    // 2. If binary params provided, overwrite parameters from binary data
    if (!binParams.empty()) {
      ModelSerializer::loadCNNParametersBinary(binParams, coreConfig, coreConfig.layersConfig);
    }

    // 3. Parse epoch history and set startingEpoch from saved training metadata
    if (json.contains("trainingMetadata")) {
      const auto& md = json.at("trainingMetadata");

      // Set startingEpoch from lastEpoch: the training loop uses 0-based epoch
      // indices, so if lastEpoch=25 was the last completed epoch, the loop
      // should resume at startingEpoch=25 (i.e., e=25..99 = 75 more epochs).
      if (md.contains("lastEpoch")) {
        ulong lastEpoch = md.at("lastEpoch").get<ulong>();

        if (lastEpoch > 0) {
          coreConfig.trainingConfig.startingEpoch = lastEpoch;
        }
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
  // Sample and input loading
  //===================================================================================================================//

  CNN::Samples<float> CNNLoader::loadSamples(const std::string& samplesFilePath, const CNN::Shape3D& inputShape,
                                             const IOConfig& ioConfig, ulong progressReports)
  {
    QFile file(QString::fromStdString(samplesFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
      throw std::runtime_error("Failed to open samples file: " + samplesFilePath);
    }

    QByteArray fileData = file.readAll();
    nlohmann::json json = nlohmann::json::parse(fileData.toStdString());

    std::string baseDir = QFileInfo(QString::fromStdString(samplesFilePath)).absolutePath().toStdString();

    const nlohmann::json& samplesArray = json.at("samples");
    size_t totalSamples = samplesArray.size();

    CNN::Samples<float> samples;
    samples.reserve(totalSamples);
    size_t idx = 0;

    for (const auto& sampleJson : samplesArray) {
      CNN::Sample<float> sample;

      // Input
      if (ioConfig.inputType == DataType::IMAGE) {
        std::string imgPath = ImageLoader::resolvePath(sampleJson.at("input").get<std::string>(), baseDir);
        std::vector<float> flatInput = ImageLoader::loadImage(
          imgPath, static_cast<int>(inputShape.c), static_cast<int>(inputShape.h), static_cast<int>(inputShape.w));
        sample.input = CNN::Input<float>(inputShape);
        sample.input.data = std::move(flatInput);
      } else {
        std::vector<float> flatInput = sampleJson.at("input").get<std::vector<float>>();

        if (flatInput.size() != inputShape.size()) {
          throw std::runtime_error("Sample input size (" + std::to_string(flatInput.size()) +
                                   ") does not match expected input shape size (" + std::to_string(inputShape.size()) +
                                   ")");
        }

        sample.input = CNN::Input<float>(inputShape);
        sample.input.data = std::move(flatInput);
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
        sample.output = sampleJson.at("output").get<CNN::Output<float>>();
      }

      samples.push_back(std::move(sample));
      ProgressBar::printLoadingProgress("Loading samples:", ++idx, totalSamples, progressReports);
    }

    return samples;
  }

  //===================================================================================================================//

  std::vector<CNN::Input<float>> CNNLoader::loadInputs(const std::string& inputFilePath, const CNN::Shape3D& inputShape,
                                                       const IOConfig& ioConfig, ulong progressReports)
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
    std::vector<CNN::Input<float>> inputs;
    inputs.reserve(totalInputs);
    size_t idx = 0;

    for (const auto& entry : inputsArray) {
      std::vector<float> flatInput;

      if (ioConfig.inputType == DataType::IMAGE) {
        std::string imgPath = ImageLoader::resolvePath(entry.get<std::string>(), baseDir);
        flatInput = ImageLoader::loadImage(imgPath, static_cast<int>(inputShape.c), static_cast<int>(inputShape.h),
                                           static_cast<int>(inputShape.w));
      } else {
        flatInput = entry.get<std::vector<float>>();
      }

      if (flatInput.size() != inputShape.size()) {
        throw std::runtime_error("Input size (" + std::to_string(flatInput.size()) +
                                 ") does not match expected input shape size (" + std::to_string(inputShape.size()) +
                                 ")");
      }

      CNN::Input<float> input(inputShape);
      input.data = std::move(flatInput);
      inputs.push_back(std::move(input));
      ProgressBar::printLoadingProgress("Loading inputs:", ++idx, totalInputs, progressReports);
    }

    return inputs;
  }

  //===================================================================================================================//

} // namespace NN_CLI
