#include "NN-CLI_ModelSerializer.hpp"

#include "NN-CLI_DataType.hpp"

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
    if (augConfig.augmentationFactor > 0)
      tcJson["augmentationFactor"] = augConfig.augmentationFactor;

    if (augConfig.balanceAugmentation)
      tcJson["balanceAugmentation"] = augConfig.balanceAugmentation;

    if (augConfig.autoClassWeights)
      tcJson["autoClassWeights"] = augConfig.autoClassWeights;

    if (augConfig.augmentationFactor > 0 || augConfig.balanceAugmentation) {
      tcJson["augmentationProbability"] = augConfig.augmentationProbability;

      nlohmann::ordered_json atJson;
      atJson["horizontalFlip"] = augConfig.transforms.horizontalFlip;
      atJson["rotation"] = augConfig.transforms.rotation;
      atJson["translation"] = augConfig.transforms.translation;
      atJson["brightness"] = augConfig.transforms.brightness;
      atJson["contrast"] = augConfig.transforms.contrast;
      atJson["gaussianNoise"] = augConfig.transforms.gaussianNoise;
      tcJson["augmentationTransforms"] = atJson;
    }
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
  //-- saveANNModel --//
  //===================================================================================================================//

  void ModelSerializer::saveANNModel(const std::string& filePath, const ANN::Core<float>& core,
                                     const ANN::CoreConfig<float>& coreConfig, const IOConfig& ioConfig,
                                     const AugmentationConfig& augConfig, const ValidationMetadata& validationMeta)
  {
    nlohmann::ordered_json json;

    json["mode"] = ANN::Mode::typeToName(core.getModeType());
    json["device"] = ANN::Device::typeToName(core.getDeviceType());
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

    json["layersConfig"] = layersArr;

    // Cost function config
    nlohmann::ordered_json cfcJson;
    cfcJson["type"] = ANN::CostFunction::typeToName(core.getCostFunctionConfig().type);

    if (!core.getCostFunctionConfig().weights.empty()) {
      cfcJson["weights"] = core.getCostFunctionConfig().weights;
    }

    json["costFunctionConfig"] = cfcJson;

    // Training config
    nlohmann::ordered_json tcJson;
    tcJson["numEpochs"] = core.getTrainingConfig().numEpochs;
    tcJson["learningRate"] = core.getTrainingConfig().learningRate;
    tcJson["batchSize"] = core.getTrainingConfig().batchSize;
    tcJson["shuffleSamples"] = core.getTrainingConfig().shuffleSamples;

    if (core.getTrainingConfig().dropoutRate > 0.0f)
      tcJson["dropoutRate"] = core.getTrainingConfig().dropoutRate;

    if (core.getTrainingConfig().optimizer.type != ANN::OptimizerType::SGD) {
      nlohmann::ordered_json optJson;
      optJson["type"] = ANN::Optimizer<float>::typeToName(core.getTrainingConfig().optimizer.type);
      optJson["beta1"] = core.getTrainingConfig().optimizer.beta1;
      optJson["beta2"] = core.getTrainingConfig().optimizer.beta2;
      optJson["epsilon"] = core.getTrainingConfig().optimizer.epsilon;
      tcJson["optimizer"] = optJson;
    }

    serializeAugConfig(tcJson, augConfig);
    json["trainingConfig"] = tcJson;

    // Test config
    nlohmann::ordered_json testConfigJson;
    testConfigJson["batchSize"] = coreConfig.testConfig.batchSize;
    json["testConfig"] = testConfigJson;

    // Training metadata
    const auto& md = core.getTrainingMetadata();
    nlohmann::ordered_json mdJson;
    mdJson["startTime"] = md.startTime;
    mdJson["endTime"] = md.endTime;
    mdJson["durationSeconds"] = md.durationSeconds;
    mdJson["durationFormatted"] = md.durationFormatted;
    mdJson["numSamples"] = md.numSamples;
    mdJson["finalLoss"] = md.finalLoss;
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

    json["mode"] = CNN::Mode::typeToName(core.getModeType());
    json["device"] = CNN::Device::typeToName(core.getDeviceType());
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
      }

      cnnLayersArr.push_back(layerJson);
    }

    json["convolutionalLayersConfig"] = cnnLayersArr;

    // Dense layers config
    nlohmann::ordered_json denseLayersArr = nlohmann::ordered_json::array();

    for (const auto& layer : core.getLayersConfig().denseLayers) {
      nlohmann::ordered_json layerJson;
      layerJson["numNeurons"] = layer.numNeurons;
      layerJson["actvFunc"] = ANN::ActvFunc::typeToName(layer.actvFuncType);
      denseLayersArr.push_back(layerJson);
    }

    json["denseLayersConfig"] = denseLayersArr;

    // Cost function config
    nlohmann::ordered_json cfcJson;
    cfcJson["type"] = CNN::CostFunction::typeToName(core.getCostFunctionConfig().type);

    if (!core.getCostFunctionConfig().weights.empty()) {
      cfcJson["weights"] = core.getCostFunctionConfig().weights;
    }

    json["costFunctionConfig"] = cfcJson;

    // Training config
    nlohmann::ordered_json tcJson;
    tcJson["numEpochs"] = core.getTrainingConfig().numEpochs;
    tcJson["learningRate"] = core.getTrainingConfig().learningRate;
    tcJson["batchSize"] = core.getTrainingConfig().batchSize;
    tcJson["shuffleSamples"] = core.getTrainingConfig().shuffleSamples;

    if (core.getTrainingConfig().dropoutRate > 0.0f)
      tcJson["dropoutRate"] = core.getTrainingConfig().dropoutRate;

    if (core.getTrainingConfig().optimizer.type != CNN::OptimizerType::SGD) {
      nlohmann::ordered_json optJson;
      optJson["type"] = CNN::Optimizer<float>::typeToName(core.getTrainingConfig().optimizer.type);
      optJson["beta1"] = core.getTrainingConfig().optimizer.beta1;
      optJson["beta2"] = core.getTrainingConfig().optimizer.beta2;
      optJson["epsilon"] = core.getTrainingConfig().optimizer.epsilon;
      tcJson["optimizer"] = optJson;
    }

    serializeAugConfig(tcJson, augConfig);
    json["trainingConfig"] = tcJson;

    // Test config
    nlohmann::ordered_json testConfigJson;
    testConfigJson["batchSize"] = coreConfig.testConfig.batchSize;
    json["testConfig"] = testConfigJson;

    // Training metadata
    const auto& md = core.getTrainingMetadata();
    nlohmann::ordered_json mdJson;
    mdJson["startTime"] = md.startTime;
    mdJson["endTime"] = md.endTime;
    mdJson["durationSeconds"] = md.durationSeconds;
    mdJson["durationFormatted"] = md.durationFormatted;
    mdJson["numSamples"] = md.numSamples;
    mdJson["finalLoss"] = md.finalLoss;
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

    if (!outputDir.exists()) {
      inputDir.mkdir("output");
    }

    QString outputPath = outputDir.filePath(QString::fromStdString(generateTrainingFilename(epochs, samples, loss)));
    return outputPath.toStdString();
  }

  //===================================================================================================================//

  std::string ModelSerializer::generateCheckpointPath(const QString& inputFilePath, ulong epoch, float loss)
  {
    QFileInfo inputInfo(inputFilePath);
    QDir inputDir = inputInfo.absoluteDir();
    QDir outputDir(inputDir.filePath("output"));

    if (!outputDir.exists()) {
      inputDir.mkdir("output");
    }

    std::ostringstream oss;
    oss << "checkpoint_E-" << epoch << "_L-" << std::fixed << std::setprecision(6) << loss << ".json";

    QString outputPath = outputDir.filePath(QString::fromStdString(oss.str()));
    return outputPath.toStdString();
  }

} // namespace NN_CLI
