#include "NN-Server_Loader.hpp"

#include "NN-CLI_ModelPackage.hpp"
#include "NN-CLI_ModelSerializer.hpp"

#include <QFile>
#include <json.hpp>

#include <stdexcept>

namespace NN_Server
{

  //===================================================================================================================//
  //-- Anonymous namespace: local helpers --//
  //===================================================================================================================//

  namespace {

    //-- Helper: read JSON from a regular file path or a .nnmodel package --//

    nlohmann::json readJsonFromPath(const std::string& path)
    {
      if (NN_CLI::ModelPackage::isPackage(path)) {
        std::string jsonStr = NN_CLI::ModelPackage::readJsonFromPackage(path);
        return nlohmann::json::parse(jsonStr);
      }

      QFile file(QString::fromStdString(path));

      if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("Failed to open config file: " + path);
      }

      QByteArray fileData = file.readAll();
      return nlohmann::json::parse(fileData.toStdString());
    }

  } // anonymous namespace

  //===================================================================================================================//
  // Network type detection
  //===================================================================================================================//

  NetworkType Loader::detectNetworkType(const std::string& configFilePath)
  {
    nlohmann::json json = readJsonFromPath(configFilePath);

    // ANN configs use "layers" for their dense layers.
    // CNN configs use "convolutionalLayers" and/or "denseLayers".
    // "inputShape" is NOT used for detection — both types can have it (e.g. image input).
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
    nlohmann::json json = readJsonFromPath(configFilePath);

    InputConfig config;

    bool isCNN = !json.contains("layers");

    if (json.contains("inputType") && json["inputType"].get<std::string>() == "image") {
      config.isImage = true;
    } else if (isCNN) {
      // CNN models default to image input even without explicit inputType
      config.isImage = true;
    }

    // Input shape (for image input — CNN uses CoreConfig.inputShape)
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
    nlohmann::json json = readJsonFromPath(configFilePath);

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
  // ANN config loading (predict-only)
  //===================================================================================================================//

  ANN::CoreConfig<float> Loader::loadConfig(const std::string& configFilePath)
  {
    nlohmann::json json;
    std::vector<char> binData;

    if (isPackage(configFilePath)) {
      auto pkg = loadPackage(configFilePath);
      json = std::move(pkg.first);
      binData = std::move(pkg.second);
    } else {
      json = readJsonFromPath(configFilePath);
    }

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

    // Load parameters — binary (package) only; legacy embedded parameters rejected
    if (!binData.empty()) {
      NN_CLI::ModelSerializer::loadANNParametersBinary(binData, coreConfig, coreConfig.layersConfig);
    } else {
      if (json.contains("parameters")) {
        throw std::runtime_error(
          "This JSON file contains embedded parameters. "
          "The embedded-parameter format is no longer supported. "
          "Server requires a .nnmodel package with separate parameter files.");
      }
      throw std::runtime_error("Config file missing parameters. Server requires a .nnmodel package for predict mode: " + configFilePath);
    }

    return coreConfig;
  }

  //===================================================================================================================//
  // CNN config loading (predict-only)
  //===================================================================================================================//

  CNN::CoreConfig<float> Loader::loadCNNConfig(const std::string& configFilePath)
  {
    nlohmann::json json;
    std::vector<char> binData;

    if (isPackage(configFilePath)) {
      auto pkg = loadPackage(configFilePath);
      json = std::move(pkg.first);
      binData = std::move(pkg.second);
    } else {
      json = readJsonFromPath(configFilePath);
    }

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

    // Load parameters — binary (package) only; legacy embedded parameters rejected
    if (!binData.empty()) {
      NN_CLI::ModelSerializer::loadCNNParametersBinary(binData, coreConfig, coreConfig.layersConfig);
    } else {
      if (json.contains("parameters")) {
        throw std::runtime_error(
          "This JSON file contains embedded parameters. "
          "The embedded-parameter format is no longer supported. "
          "Server requires a .nnmodel package with separate parameter files.");
      }
      throw std::runtime_error("CNN config file missing parameters. Server requires a .nnmodel package for predict mode: " + configFilePath);
    }

    return coreConfig;
  }

  //===================================================================================================================//
  // Package detection
  //===================================================================================================================//

  bool Loader::isPackage(const std::string& path)
  {
    return NN_CLI::ModelPackage::isPackage(path);
  }

  //===================================================================================================================//
  // Package loading
  //===================================================================================================================//

  std::pair<nlohmann::json, std::vector<char>> Loader::loadPackage(const std::string& packagePath)
  {
    std::string jsonStr = NN_CLI::ModelPackage::readJsonFromPackage(packagePath);
    nlohmann::json json = nlohmann::json::parse(jsonStr);
    std::vector<char> binData = NN_CLI::ModelPackage::readBinaryFromPackage(packagePath);
    return {std::move(json), std::move(binData)};
  }

  //===================================================================================================================//
  // ANN config loading with binary params (overload)
  //===================================================================================================================//

  ANN::CoreConfig<float> Loader::loadConfig(const std::string& configFilePath,
                                             const std::vector<char>& binParams)
  {
    if (binParams.empty()) {
      return loadConfig(configFilePath);
    }

    // Read JSON from file (not a package — binary params provided separately)
    nlohmann::json json = readJsonFromPath(configFilePath);

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

    // Load parameters from binary data
    NN_CLI::ModelSerializer::loadANNParametersBinary(binParams, coreConfig, coreConfig.layersConfig);

    return coreConfig;
  }

  //===================================================================================================================//
  // CNN config loading with binary params (overload)
  //===================================================================================================================//

  CNN::CoreConfig<float> Loader::loadCNNConfig(const std::string& configFilePath,
                                                const std::vector<char>& binParams)
  {
    if (binParams.empty()) {
      return loadCNNConfig(configFilePath);
    }

    // Read JSON from file (not a package — binary params provided separately)
    nlohmann::json json = readJsonFromPath(configFilePath);

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

    // Load parameters from binary data
    NN_CLI::ModelSerializer::loadCNNParametersBinary(binParams, coreConfig, coreConfig.layersConfig);

    return coreConfig;
  }

  //===================================================================================================================//

} // namespace NN_Server
