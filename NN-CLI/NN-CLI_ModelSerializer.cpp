#include "NN-CLI_ModelSerializer.hpp"

#include "NN-CLI_DataType.hpp"
#include "NN-CLI_Utils.hpp"

#include <QDir>
#include <QFile>
#include <QFileInfo>

#include <json.hpp>

#include <iomanip>
#include <sstream>
#include <variant>

namespace NN_CLI
{

  //===================================================================================================================//
  //-- Helper: serialize augmentation config into training config JSON --//
  //===================================================================================================================//

  static void serializeAugConfig(nlohmann::ordered_json& tcJson, const AugmentationConfig& augConfig)
  {
    tcJson["augmentationFactor"] = augConfig.augmentationFactor;
    tcJson["balanceAugmentation"] = augConfig.balanceAugmentation;
    tcJson["fullAugmentation"] = augConfig.fullAugmentation;
    tcJson["autoClassWeights"] = augConfig.autoClassWeights;
    tcJson["augmentationProbability"] = augConfig.augmentationProbability;

    nlohmann::ordered_json atJson;
    atJson["horizontalFlip"] = augConfig.transforms.horizontalFlip;
    atJson["rotation"] = augConfig.transforms.rotation;
    atJson["translation"] = augConfig.transforms.translation;
    atJson["brightness"] = augConfig.transforms.brightness;
    atJson["contrast"] = augConfig.transforms.contrast;
    atJson["gaussianNoise"] = augConfig.transforms.gaussianNoise;
    atJson["randomErasing"] = augConfig.transforms.randomErasing;
    atJson["hueShift"] = augConfig.transforms.hueShift;
    atJson["scaling"] = augConfig.transforms.scaling;

    if (augConfig.transforms.elasticDeformation.alpha > 0.0f) {
      nlohmann::ordered_json edJson;
      edJson["alpha"] = augConfig.transforms.elasticDeformation.alpha;
      edJson["sigma"] = augConfig.transforms.elasticDeformation.sigma;
      atJson["elasticDeformation"] = edJson;
    }

    tcJson["augmentationTransforms"] = atJson;
  }

  //===================================================================================================================//
  //-- Helper: serialize validation config --//
  //===================================================================================================================//

  static void serializeValidationConfig(nlohmann::ordered_json& tcJson, const AugmentationConfig& augConfig)
  {
    const auto& vc = augConfig.validationConfig;
    nlohmann::ordered_json vcJson;
    vcJson["enabled"] = vc.enabled;
    vcJson["autoSize"] = vc.autoSize;
    vcJson["size"] = vc.size;
    vcJson["checkInterval"] = vc.checkInterval;
    tcJson["validation"] = vcJson;
  }

  //===================================================================================================================//
  //-- Helper: serialize monitoring config --//
  //===================================================================================================================//

  template <typename CoreT>
  static void serializeMonitoringConfig(nlohmann::ordered_json& tcJson, const CoreT& core)
  {
    const auto& mc = core.getTrainingConfig().monitoringConfig;
    nlohmann::ordered_json mcJson;
    mcJson["enabled"] = mc.enabled;
    mcJson["checkInterval"] = mc.checkInterval;
    mcJson["patience"] = mc.patience;

    nlohmann::ordered_json metricsJson;

    nlohmann::ordered_json lsJson;
    lsJson["enabled"] = mc.metrics.lossStagnation.enabled;
    lsJson["minDelta"] = mc.metrics.lossStagnation.minDelta;
    metricsJson["lossStagnation"] = lsJson;

    nlohmann::ordered_json leJson;
    leJson["enabled"] = mc.metrics.lossExplosion.enabled;
    leJson["threshold"] = mc.metrics.lossExplosion.threshold;
    metricsJson["lossExplosion"] = leJson;

    mcJson["metrics"] = metricsJson;
    tcJson["monitoring"] = mcJson;
  }

  //===================================================================================================================//
  //-- Helper: serialize validation metadata --//
  //===================================================================================================================//

  static void serializeValidationMeta(nlohmann::ordered_json& mdJson, const ValidationMetadata& validationMeta)
  {
    if (validationMeta.enabled) {
      mdJson["numValidationSamples"] = validationMeta.numValSamples;
      mdJson["finalValidationLoss"] = validationMeta.lastValLoss;
      mdJson["bestValidationLoss"] = validationMeta.bestValLoss;
      mdJson["bestValidationEpoch"] = validationMeta.bestValEpoch;
    }
  }

  //===================================================================================================================//
  //-- Helper: serialize training config --//
  //===================================================================================================================//

  template <typename TrainingConfigT>
  static void serializeTrainingConfig(nlohmann::ordered_json& tcJson, const TrainingConfigT& tc)
  {
    tcJson["numEpochs"] = tc.numEpochs;
    tcJson["learningRate"] = tc.learningRate;
    tcJson["batchSize"] = tc.batchSize;
    tcJson["shuffleSamples"] = tc.shuffleSamples;

    tcJson["dropoutRate"] = tc.dropoutRate;

    nlohmann::ordered_json optJson;
    using OptimizerT = std::decay_t<decltype(tc.optimizer)>;
    optJson["type"] = OptimizerT::typeToName(tc.optimizer.type);
    optJson["beta1"] = tc.optimizer.beta1;
    optJson["beta2"] = tc.optimizer.beta2;
    optJson["epsilon"] = tc.optimizer.epsilon;
    tcJson["optimizer"] = optJson;
  }

  //===================================================================================================================//
  //-- Helper: serialize test config --//
  //===================================================================================================================//

  template <typename TestConfigT>
  static void serializeTestConfig(nlohmann::ordered_json& testJson, const TestConfigT& testConfig)
  {
    testJson["batchSize"] = testConfig.batchSize;
  }

  //===================================================================================================================//
  //-- Helper: serialize training metadata --//
  //===================================================================================================================//

  template <typename MetadataT>
  static void serializeTrainingMetadata(nlohmann::ordered_json& mdJson, const MetadataT& md)
  {
    mdJson["startTime"] = md.startTime;
    mdJson["endTime"] = md.endTime;
    mdJson["durationSeconds"] = md.durationSeconds;
    mdJson["durationFormatted"] = md.durationFormatted;
    mdJson["numSamples"] = md.numSamples;
    mdJson["finalLoss"] = md.finalLoss;

    if (md.lastEpoch > 0)
      mdJson["lastEpoch"] = md.lastEpoch;

    if (!md.stopReason.empty())
      mdJson["stopReason"] = md.stopReason;

    if (md.bestEpoch > 0) {
      mdJson["bestEpoch"] = md.bestEpoch;
      mdJson["bestLoss"] = md.bestLoss;
    }
  }

  //===================================================================================================================//
  //-- Helper: write JSON to file --//
  //===================================================================================================================//

  static void writeJsonToFile(const std::string& filePath, const nlohmann::ordered_json& json)
  {
    QFile file(QString::fromStdString(filePath));

    if (!file.open(QIODevice::WriteOnly)) {
      throw std::runtime_error("Failed to open file for writing: " + filePath);
    }

    std::string jsonStr = json.dump(4);
    file.write(jsonStr.c_str());
    file.close();
  }

  //===================================================================================================================//
  //-- saveModel --//
  //===================================================================================================================//

  void ModelSerializer::saveModel(const std::string& filePath, const ANN::Core<float>& core,
                                     const ANN::CoreConfig<float>& coreConfig, const IOConfig& ioConfig,
                                     const AugmentationConfig& augConfig, const ValidationMetadata& validationMeta)
  {
    nlohmann::ordered_json json;

    json["mode"] = Common::Mode::typeToName(core.getModeType());
    json["device"] = ::Device::typeToName(core.getDeviceType());
    json["numThreads"] = core.getNumThreads();
    json["numGPUs"] = core.getNumGPUs();

    // NN-CLI settings
    json["progressReports"] = coreConfig.progressReports;
    json["saveModelInterval"] = ioConfig.saveModelInterval;

    // I/O types
    json["inputType"] = dataTypeToString(ioConfig.inputType);
    json["outputType"] = dataTypeToString(ioConfig.outputType);

    if (ioConfig.hasInputShape()) {
      nlohmann::ordered_json isJson;
      isJson["c"] = ioConfig.inputC;
      isJson["h"] = ioConfig.inputH;
      isJson["w"] = ioConfig.inputW;
      json["inputShape"] = isJson;
    }

    if (ioConfig.hasOutputShape()) {
      nlohmann::ordered_json osJson;
      osJson["c"] = ioConfig.outputC;
      osJson["h"] = ioConfig.outputH;
      osJson["w"] = ioConfig.outputW;
      json["outputShape"] = osJson;
    }

    // Layers config
    nlohmann::ordered_json layersArr = nlohmann::ordered_json::array();

    for (const auto& layer : core.getLayersConfig()) {
      nlohmann::ordered_json layerJson;
      layerJson["numNeurons"] = layer.numNeurons;
      layerJson["actvFunc"] = ANN::ActvFunc::typeToName(layer.actvFuncType);
      layersArr.push_back(layerJson);
    }

    json["layers"] = layersArr;

    // Cost function config
    nlohmann::ordered_json cfcJson;
    cfcJson["type"] = Common::CostFunction::typeToName(core.getCostFunctionConfig().type);

    if (!core.getCostFunctionConfig().weights.empty()) {
      cfcJson["weights"] = core.getCostFunctionConfig().weights;
    }

    json["costFunction"] = cfcJson;

    // Training config
    nlohmann::ordered_json tcJson;
    serializeTrainingConfig(tcJson, core.getTrainingConfig());
    serializeAugConfig(tcJson, augConfig);
    serializeValidationConfig(tcJson, augConfig);
    serializeMonitoringConfig(tcJson, core);
    json["training"] = tcJson;

    // Test config
    nlohmann::ordered_json testJson;
    serializeTestConfig(testJson, coreConfig.testConfig);
    json["test"] = testJson;

    // Training metadata
    const auto& md = core.getTrainingMetadata();
    nlohmann::ordered_json mdJson;
    serializeTrainingMetadata(mdJson, md);
    serializeValidationMeta(mdJson, validationMeta);
    json["trainingMetadata"] = mdJson;

    // Parameters
    nlohmann::ordered_json paramsJson;
    paramsJson["weights"] = core.getParameters().weights;
    paramsJson["biases"] = core.getParameters().biases;
    json["parameters"] = paramsJson;

    writeJsonToFile(filePath, json);
  }

  //===================================================================================================================//
  //-- saveCNNModel --//
  //===================================================================================================================//

  void ModelSerializer::saveCNNModel(const std::string& filePath, const CNN::Core<float>& core,
                                     const CNN::CoreConfig<float>& coreConfig, const IOConfig& ioConfig,
                                     const AugmentationConfig& augConfig, const ValidationMetadata& validationMeta)
  {
    nlohmann::ordered_json json;

    json["mode"] = Common::Mode::typeToName(core.getModeType());
    json["device"] = ::Device::typeToName(core.getDeviceType());
    json["numThreads"] = core.getNumThreads();
    json["numGPUs"] = core.getNumGPUs();

    // NN-CLI settings
    json["progressReports"] = coreConfig.progressReports;
    json["saveModelInterval"] = ioConfig.saveModelInterval;

    // I/O types
    json["inputType"] = dataTypeToString(ioConfig.inputType);
    json["outputType"] = dataTypeToString(ioConfig.outputType);

    // Input shape
    const auto& shape = core.getInputShape();
    nlohmann::ordered_json shapeJson;
    shapeJson["c"] = shape.c;
    shapeJson["h"] = shape.h;
    shapeJson["w"] = shape.w;
    json["inputShape"] = shapeJson;

    // Output shape
    if (ioConfig.hasOutputShape()) {
      nlohmann::ordered_json osJson;
      osJson["c"] = ioConfig.outputC;
      osJson["h"] = ioConfig.outputH;
      osJson["w"] = ioConfig.outputW;
      json["outputShape"] = osJson;
    }

    // CNN layers config
    nlohmann::ordered_json cnnLayersArr = nlohmann::ordered_json::array();

    for (const auto& layer : core.getLayersConfig().cnnLayers) {
      nlohmann::ordered_json layerJson;

      switch (layer.type) {
      case CNN::LayerType::CONV: {
        const auto& conv = std::get<CNN::ConvLayerConfig>(layer.config);
        layerJson["type"] = "conv";
        layerJson["numFilters"] = conv.numFilters;
        layerJson["filterH"] = conv.filterH;
        layerJson["filterW"] = conv.filterW;
        layerJson["strideY"] = conv.strideY;
        layerJson["strideX"] = conv.strideX;
        layerJson["slidingStrategy"] = CNN::SlidingStrategy::typeToName(conv.slidingStrategy);
        break;
      }

      case CNN::LayerType::RELU:
        layerJson["type"] = "relu";
        break;
      case CNN::LayerType::POOL: {
        const auto& pool = std::get<CNN::PoolLayerConfig>(layer.config);
        layerJson["type"] = "pool";
        layerJson["poolType"] = CNN::PoolType::typeToName(pool.poolType);
        layerJson["poolH"] = pool.poolH;
        layerJson["poolW"] = pool.poolW;
        layerJson["strideY"] = pool.strideY;
        layerJson["strideX"] = pool.strideX;
        break;
      }

      case CNN::LayerType::INSTANCENORM: {
        const auto& bn = std::get<CNN::NormLayerConfig>(layer.config);
        layerJson["type"] = "instancenorm";
        layerJson["epsilon"] = bn.epsilon;
        layerJson["momentum"] = bn.momentum;
        break;
      }

      case CNN::LayerType::BATCHNORM: {
        const auto& bn = std::get<CNN::NormLayerConfig>(layer.config);
        layerJson["type"] = "batchnorm";
        layerJson["epsilon"] = bn.epsilon;
        layerJson["momentum"] = bn.momentum;
        break;
      }

      case CNN::LayerType::GLOBALAVGPOOL:
        layerJson["type"] = "globalavgpool";
        break;
      case CNN::LayerType::GLOBALDUALPOOL:
        layerJson["type"] = "globaldualpool";
        break;
      case CNN::LayerType::FLATTEN:
        layerJson["type"] = "flatten";
        break;
      case CNN::LayerType::RESIDUAL_START:
        layerJson["type"] = "residual_start";
        break;
      case CNN::LayerType::RESIDUAL_END:
        layerJson["type"] = "residual_end";
        break;

      default: {
        std::ostringstream oss;
        oss << "Unknown CNN layer type in serializer: " << static_cast<int>(layer.type);
        throw std::runtime_error(oss.str());
      }
      }

      cnnLayersArr.push_back(layerJson);
    }

    json["convolutionalLayers"] = cnnLayersArr;

    // Dense layers config
    nlohmann::ordered_json denseLayersArr = nlohmann::ordered_json::array();

    for (const auto& layer : core.getLayersConfig().denseLayers) {
      nlohmann::ordered_json layerJson;
      layerJson["numNeurons"] = layer.numNeurons;
      layerJson["actvFunc"] = ANN::ActvFunc::typeToName(layer.actvFuncType);
      denseLayersArr.push_back(layerJson);
    }

    json["denseLayers"] = denseLayersArr;

    // Cost function config
    nlohmann::ordered_json cfcJson;
    cfcJson["type"] = Common::CostFunction::typeToName(core.getCostFunctionConfig().type);

    if (!core.getCostFunctionConfig().weights.empty()) {
      cfcJson["weights"] = core.getCostFunctionConfig().weights;
    }

    json["costFunction"] = cfcJson;

    // Training config
    nlohmann::ordered_json tcJson;
    serializeTrainingConfig(tcJson, core.getTrainingConfig());
    serializeAugConfig(tcJson, augConfig);
    serializeValidationConfig(tcJson, augConfig);
    serializeMonitoringConfig(tcJson, core);
    json["training"] = tcJson;

    // Test config
    nlohmann::ordered_json testJson;
    serializeTestConfig(testJson, coreConfig.testConfig);
    json["test"] = testJson;

    // Training metadata
    const auto& md = core.getTrainingMetadata();
    nlohmann::ordered_json mdJson;
    serializeTrainingMetadata(mdJson, md);
    serializeValidationMeta(mdJson, validationMeta);
    json["trainingMetadata"] = mdJson;

    // Parameters
    nlohmann::ordered_json paramsJson;

    // Conv parameters
    nlohmann::ordered_json convArr = nlohmann::ordered_json::array();

    for (const auto& cp : core.getParameters().convParams) {
      nlohmann::ordered_json cpJson;
      cpJson["numFilters"] = cp.numFilters;
      cpJson["inputC"] = cp.inputC;
      cpJson["filterH"] = cp.filterH;
      cpJson["filterW"] = cp.filterW;
      cpJson["filters"] = cp.filters;
      cpJson["biases"] = cp.biases;
      convArr.push_back(cpJson);
    }

    paramsJson["convolutional"] = convArr;

    // Norm parameters
    if (!core.getParameters().normParams.empty()) {
      nlohmann::ordered_json normArr = nlohmann::ordered_json::array();

      for (const auto& bp : core.getParameters().normParams) {
        nlohmann::ordered_json bpJson;
        bpJson["numChannels"] = bp.numChannels;
        bpJson["gamma"] = bp.gamma;
        bpJson["beta"] = bp.beta;
        bpJson["runningMean"] = bp.runningMean;
        bpJson["runningVar"] = bp.runningVar;
        normArr.push_back(bpJson);
      }

      paramsJson["instancenorm"] = normArr;
    }

    // Residual projection parameters
    if (!core.getParameters().residualParams.empty()) {
      nlohmann::ordered_json resArr = nlohmann::ordered_json::array();

      for (const auto& rp : core.getParameters().residualParams) {
        nlohmann::ordered_json rpJson;
        rpJson["inC"] = rp.inC;
        rpJson["outC"] = rp.outC;
        rpJson["weights"] = rp.weights;
        rpJson["biases"] = rp.biases;
        resArr.push_back(rpJson);
      }

      paramsJson["residual"] = resArr;
    }

    // Dense parameters
    nlohmann::ordered_json denseParamsJson;
    denseParamsJson["weights"] = core.getParameters().denseParams.weights;
    denseParamsJson["biases"] = core.getParameters().denseParams.biases;
    paramsJson["dense"] = denseParamsJson;

    json["parameters"] = paramsJson;

    writeJsonToFile(filePath, json);
  }

  //===================================================================================================================//
  //-- Output path helpers --//
  //===================================================================================================================//

  std::string ModelSerializer::generateTrainingFilename(ulong epochs, ulong samples, float loss)
  {
    std::ostringstream oss;
    oss << "trained_E-" << epochs << "_S-" << samples << "_L-" << std::fixed << std::setprecision(6) << loss << ".json";
    return oss.str();
  }

  //===================================================================================================================//

  std::string ModelSerializer::generateDefaultOutputPath(const QString& inputFilePath, ulong epochs, ulong samples,
                                                         float loss)
  {
    QFileInfo inputInfo(inputFilePath);
    QDir inputDir = inputInfo.absoluteDir();
    QDir outputDir(inputDir.filePath("output"));

    NN_CLI::ensureOutputDir(inputDir.filePath("output"));

    QString outputPath = outputDir.filePath(QString::fromStdString(generateTrainingFilename(epochs, samples, loss)));
    return outputPath.toStdString();
  }

  //===================================================================================================================//

  std::string ModelSerializer::generateCheckpointPath(const QString& inputFilePath, ulong epoch, float loss)
  {
    QFileInfo inputInfo(inputFilePath);
    QDir inputDir = inputInfo.absoluteDir();
    QDir outputDir(inputDir.filePath("output"));

    NN_CLI::ensureOutputDir(inputDir.filePath("output"));

    std::ostringstream oss;
    oss << "checkpoint_E-" << epoch << "_L-" << std::fixed << std::setprecision(6) << loss << ".json";

    QString outputPath = outputDir.filePath(QString::fromStdString(oss.str()));
    return outputPath.toStdString();
  }

  //===================================================================================================================//

  std::string ModelSerializer::generateBestModelPath(const QString& inputFilePath)
  {
    QFileInfo inputInfo(inputFilePath);
    QDir inputDir = inputInfo.absoluteDir();
    QDir outputDir(inputDir.filePath("output"));

    NN_CLI::ensureOutputDir(inputDir.filePath("output"));

    QString outputPath = outputDir.filePath("best_model.json");
    return outputPath.toStdString();
  }

} // namespace NN_CLI
