#include "NN-Server_Loader.hpp"

#include <QFile>
#include <json.hpp>

#include <stdexcept>

namespace NN_Server
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

    //  configs use "layers" for their dense layers.
    // CNN configs use "convolutionalLayers" and/or "denseLayers".
    // "inputShape" is NOT used for detection — both types can have it (e.g.  image input).
    if (json.contains("layers")) {
return NetworkType::ANN;
    }

    return NetworkType::CNN;
  }

  //===================================================================================================================//
  // Input configuration loading
  //===================================================================================================================//

  InputConfig Loader::loadInputConfig(const std::string& configFilePath)
  {
    QFile file(QString::fromStdString(configFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
      throw std::runtime_error("Failed to open config file: " + configFilePath);
    }

    QByteArray fileData = file.readAll();
    nlohmann::json json = nlohmann::json::parse(fileData.toStdString());

    InputConfig config;

    bool isCNN = !json.contains("layers");

    if (json.contains("inputType") && json["inputType"].get<std::string>() == "image") {
      config.isImage = true;
    } else if (isCNN) {
      // CNN models default to image input even without explicit inputType
      config.isImage = true;
    }

    // Input shape (for  image input — CNN uses CoreConfig.inputShape)
    if (json.contains("inputShape")) {
      const auto& shape = json["inputShape"];
      config.c = shape.at("c").get<ulong>();
      config.h = shape.at("h").get<ulong>();
      config.w = shape.at("w").get<ulong>();
    } else if (config.isImage) {
      throw std::runtime_error("Model has inputType \"image\" but is missing the required \"inputShape\" field.");
    }

    return config;
  }

  //===================================================================================================================//
  // Output configuration loading
  //===================================================================================================================//

  OutputConfig Loader::loadOutputConfig(const std::string& configFilePath)
  {
    QFile file(QString::fromStdString(configFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
      throw std::runtime_error("Failed to open config file: " + configFilePath);
    }

    QByteArray fileData = file.readAll();
    nlohmann::json json = nlohmann::json::parse(fileData.toStdString());

    OutputConfig config;

    if (json.contains("outputType") && json["outputType"].get<std::string>() == "image") {
      config.isImage = true;

      if (!json.contains("outputShape")) {
        throw std::runtime_error("Model has outputType \"image\" but is missing the required \"outputShape\" field.");
      }

      const auto& shape = json["outputShape"];
      config.c = shape.at("c").get<ulong>();
      config.h = shape.at("h").get<ulong>();
      config.w = shape.at("w").get<ulong>();
    }

    return config;
  }

  //===================================================================================================================//
  //  config loading (predict-only)
  //===================================================================================================================//

  ANN::CoreConfig<float> Loader::loadConfig(const std::string& configFilePath)
  {
    QFile file(QString::fromStdString(configFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
      throw std::runtime_error("Failed to open config file: " + configFilePath);
    }

    QByteArray fileData = file.readAll();
    nlohmann::json json = nlohmann::json::parse(fileData.toStdString());

    ANN::CoreConfig<float> coreConfig;

    // Always predict mode for the server
    coreConfig.modeType = Common::ModeType::PREDICT;

    if (json.contains("device")) {
      coreConfig.deviceType = Common::Device::nameToType(json.at("device").get<std::string>());
    } else {
      coreConfig.deviceType = Common::DeviceType::CPU;
    }

    if (json.contains("numThreads"))
      coreConfig.numThreads = json.at("numThreads").get<int>();

    if (json.contains("numGPUs"))
      coreConfig.numGPUs = json.at("numGPUs").get<int>();

    if (!json.contains("layers")) {
      throw std::runtime_error("Config file missing 'layers': " + configFilePath);
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

    if (!json.contains("parameters")) {
      throw std::runtime_error("Config file missing 'parameters' required for predict mode: " + configFilePath);
    }

    const auto& p = json.at("parameters");
    coreConfig.parameters.weights = p.at("weights").get<ANN::Tensor3D<float>>();
    coreConfig.parameters.biases = p.at("biases").get<ANN::Tensor2D<float>>();

    return coreConfig;
  }

  //===================================================================================================================//
  // CNN config loading (predict-only)
  //===================================================================================================================//

  CNN::CoreConfig<float> Loader::loadCNNConfig(const std::string& configFilePath)
  {
    QFile file(QString::fromStdString(configFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
      throw std::runtime_error("Failed to open config file: " + configFilePath);
    }

    QByteArray fileData = file.readAll();
    nlohmann::json json = nlohmann::json::parse(fileData.toStdString());

    CNN::CoreConfig<float> coreConfig;

    // Always predict mode for the server
    coreConfig.modeType = Common::ModeType::PREDICT;

    if (json.contains("device")) {
      coreConfig.deviceType = Common::Device::nameToType(json.at("device").get<std::string>());
    } else {
      coreConfig.deviceType = Common::DeviceType::CPU;
    }

    if (json.contains("numThreads"))
      coreConfig.numThreads = json.at("numThreads").get<int>();

    if (json.contains("numGPUs"))
      coreConfig.numGPUs = json.at("numGPUs").get<int>();

    // Input shape (required for CNN)
    if (!json.contains("inputShape")) {
      throw std::runtime_error("CNN config file missing 'inputShape': " + configFilePath);
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

    // Parameters (required for predict mode)
    if (!json.contains("parameters")) {
      throw std::runtime_error("CNN config file missing 'parameters' required for predict mode: " + configFilePath);
    }

    const auto& paramsJson = json.at("parameters");

    if (paramsJson.contains("convolutional")) {
      for (const auto& convJson : paramsJson.at("convolutional")) {
        CNN::ConvParameters<float> cp;
        cp.numFilters = convJson.at("numFilters").get<ulong>();
        cp.inputC = convJson.at("inputC").get<ulong>();
        cp.filterH = convJson.at("filterH").get<ulong>();
        cp.filterW = convJson.at("filterW").get<ulong>();
        cp.filters = convJson.at("filters").get<std::vector<float>>();
        cp.biases = convJson.at("biases").get<std::vector<float>>();
        coreConfig.parameters.convParams.push_back(std::move(cp));
      }
    }

    if (paramsJson.contains("instancenorm")) {
      for (const auto& normJson : paramsJson.at("instancenorm")) {
        CNN::NormParameters<float> bp;
        bp.numChannels = normJson.at("numChannels").get<ulong>();
        bp.gamma = normJson.at("gamma").get<std::vector<float>>();
        bp.beta = normJson.at("beta").get<std::vector<float>>();
        bp.runningMean = normJson.at("runningMean").get<std::vector<float>>();
        bp.runningVar = normJson.at("runningVar").get<std::vector<float>>();
        coreConfig.parameters.normParams.push_back(std::move(bp));
      }
    }

    if (paramsJson.contains("residual")) {
      for (const auto& resJson : paramsJson.at("residual")) {
        CNN::ResidualParameters<float> rp;
        rp.inC = resJson.at("inC").get<ulong>();
        rp.outC = resJson.at("outC").get<ulong>();
        rp.weights = resJson.at("weights").get<std::vector<float>>();
        rp.biases = resJson.at("biases").get<std::vector<float>>();
        coreConfig.parameters.residualParams.push_back(std::move(rp));
      }
    }

    if (paramsJson.contains("dense")) {
      const auto& denseJson = paramsJson.at("dense");
      coreConfig.parameters.denseParams.weights = denseJson.at("weights").get<ANN::Tensor3D<float>>();
      coreConfig.parameters.denseParams.biases = denseJson.at("biases").get<ANN::Tensor2D<float>>();
    }

    return coreConfig;
  }

  //===================================================================================================================//

} // namespace NN_Server
