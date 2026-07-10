#include "NN-CLI_ModelSerializer.hpp"
#include "NN-CLI_ModelSerializerDetail.hpp"

#include <cstring>
#include <fstream>
#include <sstream>
#include <stack>
#include <variant>

#include "CNN_SlidingStrategy.hpp"
#include "NN-CLI_DataType.hpp"
#include "NN-CLI_ModelPackage.hpp"

namespace NN_CLI
{

  //===================================================================================================================//
  //-- Helper: serialize CNN parameters to binary buffer --//
  //===================================================================================================================//

  static std::vector<char> serializeCNNParametersBinary(const CNN::Core<float>& core)
  {
    std::vector<char> buffer;
    writeBinaryHeader(buffer, BINARY_MODEL_CNN);

    const auto& params = core.getParameters();

    // Conv parameters: CONV_FILTERS + CONV_BIASES per layer
    for (size_t i = 0; i < params.convParams.size(); ++i) {
      uint32_t layerIdx = static_cast<uint32_t>(i);
      const auto& cp = params.convParams[i];

      writeBlockToBuffer(buffer, BLOCK_CONV_FILTERS, layerIdx, 1, static_cast<uint32_t>(cp.filters.size()), 0, 0,
                         cp.filters);

      writeBlockToBuffer(buffer, BLOCK_CONV_BIASES, layerIdx, 1, static_cast<uint32_t>(cp.biases.size()), 0, 0,
                         cp.biases);
    }

    // Norm parameters: GAMMA + BETA + RUNNING_MEAN + RUNNING_VAR per layer
    for (size_t i = 0; i < params.normParams.size(); ++i) {
      uint32_t layerIdx = static_cast<uint32_t>(i);
      const auto& np = params.normParams[i];

      writeBlockToBuffer(buffer, BLOCK_NORM_GAMMA, layerIdx, 1, static_cast<uint32_t>(np.gamma.size()), 0, 0, np.gamma);
      writeBlockToBuffer(buffer, BLOCK_NORM_BETA, layerIdx, 1, static_cast<uint32_t>(np.beta.size()), 0, 0, np.beta);
      writeBlockToBuffer(buffer, BLOCK_NORM_RUNNING_MEAN, layerIdx, 1, static_cast<uint32_t>(np.runningMean.size()), 0,
                         0, np.runningMean);
      writeBlockToBuffer(buffer, BLOCK_NORM_RUNNING_VAR, layerIdx, 1, static_cast<uint32_t>(np.runningVar.size()), 0, 0,
                         np.runningVar);
    }

    // Residual parameters: WEIGHTS + BIASES per layer
    for (size_t i = 0; i < params.residualParams.size(); ++i) {
      uint32_t layerIdx = static_cast<uint32_t>(i);
      const auto& rp = params.residualParams[i];

      writeBlockToBuffer(buffer, BLOCK_RESIDUAL_WEIGHTS, layerIdx, 1, static_cast<uint32_t>(rp.weights.size()), 0, 0,
                         rp.weights);
      writeBlockToBuffer(buffer, BLOCK_RESIDUAL_BIASES, layerIdx, 1, static_cast<uint32_t>(rp.biases.size()), 0, 0,
                         rp.biases);
    }

    // Dense parameters: ANN_WEIGHTS + ANN_BIASES (delegated to ANN structure)
    const auto& denseParams = params.denseParams;

    for (size_t i = 0; i < denseParams.weights.size(); ++i) {
      uint32_t layerIdx = static_cast<uint32_t>(i);

      const auto& weightMatrix = denseParams.weights[i];
      uint32_t dim0 = static_cast<uint32_t>(weightMatrix.size());
      uint32_t dim1 = (dim0 > 0) ? static_cast<uint32_t>(weightMatrix[0].size()) : 0u;

      std::vector<float> flatWeights = flattenWeights(weightMatrix);
      writeBlockToBuffer(buffer, BLOCK_ANN_WEIGHTS, layerIdx, 2, dim0, dim1, 0, flatWeights);

      const auto& biasVec = denseParams.biases[i];
      uint32_t numBiases = static_cast<uint32_t>(biasVec.size());
      std::vector<float> biasData(biasVec.begin(), biasVec.end());
      writeBlockToBuffer(buffer, BLOCK_ANN_BIASES, layerIdx, 1, numBiases, 0, 0, biasData);
    }

    return buffer;
  }

  //===================================================================================================================//
  //-- saveCNNParametersBinary --//
  //===================================================================================================================//

  void ModelSerializer::saveCNNParametersBinary(const std::string& binPath, const CNN::Core<float>& core)
  {
    std::vector<char> buffer = serializeCNNParametersBinary(core);

    std::ofstream ofs(binPath, std::ios::binary);

    if (!ofs) {
      throw std::runtime_error("Failed to open binary parameter file for writing: " + binPath);
    }

    ofs.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    ofs.close();
  }

  //===================================================================================================================//
  //-- loadCNNParametersBinary --//
  //===================================================================================================================//

  void ModelSerializer::loadCNNParametersBinary(const std::vector<char>& data, CNN::CoreConfig<float>& config,
                                                const CNN::LayersConfig& layersConfig)
  {
    // Validate minimum size
    if (data.size() < BINARY_HEADER_SIZE) {
      throw std::runtime_error("Binary parameter data too small for header");
    }

    const char* ptr = data.data();

    // Validate magic
    uint32_t magic = readU32LE(ptr + 0);

    if (magic != BINARY_MAGIC) {
      throw std::runtime_error("Invalid binary parameter magic");
    }

    // Validate header size
    uint16_t headerSize = readU16LE(ptr + 4);

    if (headerSize != BINARY_HEADER_SIZE) {
      throw std::runtime_error("Unsupported binary header size");
    }

    // Validate version
    uint8_t version = static_cast<uint8_t>(ptr[6]);

    if (version != BINARY_VERSION) {
      throw std::runtime_error("Unsupported binary parameter version");
    }

    // Validate model type (CNN)
    uint8_t modelType = static_cast<uint8_t>(ptr[8]);

    if (modelType != BINARY_MODEL_CNN) {
      throw std::runtime_error("Binary parameter data is not a CNN model");
    }

    size_t pos = BINARY_HEADER_SIZE;

    // Precompute metadata by walking the layer config with running shape tracking.
    // This gives us both the counts AND the shape metadata for each parameter set.
    struct ConvMeta {
        ulong numFilters;
        ulong inputC;
        ulong filterH;
        ulong filterW;
    };

    struct NormMeta {
        ulong numChannels;
    };

    struct ResidualMeta {
        ulong inC;
        ulong outC;
    };

    std::vector<ConvMeta> convMetaVec;
    std::vector<NormMeta> normMetaVec;
    std::vector<ResidualMeta> residualMetaVec;

    {
      CNN::Shape3D currentShape = config.inputShape;
      std::stack<CNN::Shape3D> residualShapeStack;

      for (const auto& layer : layersConfig.cnnLayers) {
        switch (layer.type) {
        case CNN::LayerType::CONV: {
          const auto& conv = std::get<CNN::ConvLayerConfig>(layer.config);
          ulong padY = CNN::SlidingStrategy::computePadding(conv.filterH, conv.slidingStrategy);
          ulong padX = CNN::SlidingStrategy::computePadding(conv.filterW, conv.slidingStrategy);

          ConvMeta meta;
          meta.numFilters = conv.numFilters;
          meta.inputC = currentShape.c;
          meta.filterH = conv.filterH;
          meta.filterW = conv.filterW;
          convMetaVec.push_back(meta);

          ulong outH = (currentShape.h + 2 * padY - conv.filterH) / conv.strideY + 1;
          ulong outW = (currentShape.w + 2 * padX - conv.filterW) / conv.strideX + 1;
          currentShape = {conv.numFilters, outH, outW};
          break;
        }

        case CNN::LayerType::POOL: {
          const auto& pool = std::get<CNN::PoolLayerConfig>(layer.config);
          ulong outH = (currentShape.h - pool.poolH) / pool.strideY + 1;
          ulong outW = (currentShape.w - pool.poolW) / pool.strideX + 1;
          currentShape = {currentShape.c, outH, outW};
          break;
        }

        case CNN::LayerType::INSTANCENORM:
        case CNN::LayerType::BATCHNORM: {
          NormMeta meta;
          meta.numChannels = currentShape.c;
          normMetaVec.push_back(meta);
          break;
        }

        case CNN::LayerType::GLOBALAVGPOOL:
          currentShape = {currentShape.c, 1, 1};
          break;

        case CNN::LayerType::GLOBALDUALPOOL:
          currentShape = {currentShape.c * 2, 1, 1};
          break;

        case CNN::LayerType::RESIDUAL_START:
          residualShapeStack.push(currentShape);
          break;

        case CNN::LayerType::RESIDUAL_END: {
          CNN::Shape3D skipShape = residualShapeStack.top();
          residualShapeStack.pop();

          if (skipShape.c != currentShape.c) {
            ResidualMeta meta;
            meta.inC = skipShape.c;
            meta.outC = currentShape.c;
            residualMetaVec.push_back(meta);
          }

          break;
        }

        case CNN::LayerType::RELU:
        case CNN::LayerType::FLATTEN:
          break;
        }
      }
    }

    size_t numConvLayers = convMetaVec.size();
    size_t numNormLayers = normMetaVec.size();
    size_t numResidualLayers = residualMetaVec.size();

    //-- Read conv parameters --//

    config.parameters.convParams.resize(numConvLayers);

    for (size_t i = 0; i < numConvLayers; ++i) {
      // CONV_FILTERS
      if (pos + BLOCK_HEADER_SIZE > data.size()) {
        throw std::runtime_error("Unexpected end of binary parameter data");
      }

      const char* blockPtr = data.data() + pos;
      uint8_t blockType = static_cast<uint8_t>(blockPtr[0]);
      uint32_t dataSize = readU32LE(blockPtr + 18);

      if (blockType != BLOCK_CONV_FILTERS) {
        throw std::runtime_error("Expected CONV_FILTERS block");
      }

      if (pos + BLOCK_HEADER_SIZE + dataSize > data.size()) {
        throw std::runtime_error("Conv filter data exceeds buffer");
      }

      config.parameters.convParams[i].filters = readFloatVector(blockPtr + BLOCK_HEADER_SIZE, dataSize);
      pos += BLOCK_HEADER_SIZE + dataSize;

      // CONV_BIASES
      if (pos + BLOCK_HEADER_SIZE > data.size()) {
        throw std::runtime_error("Unexpected end of binary parameter data");
      }

      blockPtr = data.data() + pos;
      blockType = static_cast<uint8_t>(blockPtr[0]);
      dataSize = readU32LE(blockPtr + 18);

      if (blockType != BLOCK_CONV_BIASES) {
        throw std::runtime_error("Expected CONV_BIASES block");
      }

      if (pos + BLOCK_HEADER_SIZE + dataSize > data.size()) {
        throw std::runtime_error("Conv bias data exceeds buffer");
      }

      config.parameters.convParams[i].biases = readFloatVector(blockPtr + BLOCK_HEADER_SIZE, dataSize);
      pos += BLOCK_HEADER_SIZE + dataSize;

      // Set conv metadata from precomputed shape
      config.parameters.convParams[i].numFilters = convMetaVec[i].numFilters;
      config.parameters.convParams[i].inputC = convMetaVec[i].inputC;
      config.parameters.convParams[i].filterH = convMetaVec[i].filterH;
      config.parameters.convParams[i].filterW = convMetaVec[i].filterW;
    }

    //-- Read norm parameters --//

    config.parameters.normParams.resize(numNormLayers);

    for (size_t i = 0; i < numNormLayers; ++i) {
      // NORM_GAMMA
      if (pos + BLOCK_HEADER_SIZE > data.size()) {
        throw std::runtime_error("Unexpected end of binary parameter data");
      }

      const char* blockPtr = data.data() + pos;
      uint8_t blockType = static_cast<uint8_t>(blockPtr[0]);
      uint32_t dataSize = readU32LE(blockPtr + 18);

      if (blockType != BLOCK_NORM_GAMMA) {
        throw std::runtime_error("Expected NORM_GAMMA block");
      }

      if (pos + BLOCK_HEADER_SIZE + dataSize > data.size()) {
        throw std::runtime_error("Norm gamma data exceeds buffer");
      }

      config.parameters.normParams[i].gamma = readFloatVector(blockPtr + BLOCK_HEADER_SIZE, dataSize);
      pos += BLOCK_HEADER_SIZE + dataSize;

      // NORM_BETA
      if (pos + BLOCK_HEADER_SIZE > data.size()) {
        throw std::runtime_error("Unexpected end of binary parameter data");
      }

      blockPtr = data.data() + pos;
      blockType = static_cast<uint8_t>(blockPtr[0]);
      dataSize = readU32LE(blockPtr + 18);

      if (blockType != BLOCK_NORM_BETA) {
        throw std::runtime_error("Expected NORM_BETA block");
      }

      if (pos + BLOCK_HEADER_SIZE + dataSize > data.size()) {
        throw std::runtime_error("Norm beta data exceeds buffer");
      }

      config.parameters.normParams[i].beta = readFloatVector(blockPtr + BLOCK_HEADER_SIZE, dataSize);
      pos += BLOCK_HEADER_SIZE + dataSize;

      // NORM_RUNNING_MEAN
      if (pos + BLOCK_HEADER_SIZE > data.size()) {
        throw std::runtime_error("Unexpected end of binary parameter data");
      }

      blockPtr = data.data() + pos;
      blockType = static_cast<uint8_t>(blockPtr[0]);
      dataSize = readU32LE(blockPtr + 18);

      if (blockType != BLOCK_NORM_RUNNING_MEAN) {
        throw std::runtime_error("Expected NORM_RUNNING_MEAN block");
      }

      if (pos + BLOCK_HEADER_SIZE + dataSize > data.size()) {
        throw std::runtime_error("Norm running mean data exceeds buffer");
      }

      config.parameters.normParams[i].runningMean = readFloatVector(blockPtr + BLOCK_HEADER_SIZE, dataSize);
      pos += BLOCK_HEADER_SIZE + dataSize;

      // NORM_RUNNING_VAR
      if (pos + BLOCK_HEADER_SIZE > data.size()) {
        throw std::runtime_error("Unexpected end of binary parameter data");
      }

      blockPtr = data.data() + pos;
      blockType = static_cast<uint8_t>(blockPtr[0]);
      dataSize = readU32LE(blockPtr + 18);

      if (blockType != BLOCK_NORM_RUNNING_VAR) {
        throw std::runtime_error("Expected NORM_RUNNING_VAR block");
      }

      if (pos + BLOCK_HEADER_SIZE + dataSize > data.size()) {
        throw std::runtime_error("Norm running var data exceeds buffer");
      }

      config.parameters.normParams[i].runningVar = readFloatVector(blockPtr + BLOCK_HEADER_SIZE, dataSize);
      pos += BLOCK_HEADER_SIZE + dataSize;

      // Set norm metadata from precomputed shape
      config.parameters.normParams[i].numChannels = normMetaVec[i].numChannels;
    }

    //-- Read residual parameters --//

    config.parameters.residualParams.resize(numResidualLayers);

    for (size_t i = 0; i < numResidualLayers; ++i) {
      // RESIDUAL_WEIGHTS
      if (pos + BLOCK_HEADER_SIZE > data.size()) {
        throw std::runtime_error("Unexpected end of binary parameter data");
      }

      const char* blockPtr = data.data() + pos;
      uint8_t blockType = static_cast<uint8_t>(blockPtr[0]);
      uint32_t dataSize = readU32LE(blockPtr + 18);

      if (blockType != BLOCK_RESIDUAL_WEIGHTS) {
        throw std::runtime_error("Expected RESIDUAL_WEIGHTS block");
      }

      if (pos + BLOCK_HEADER_SIZE + dataSize > data.size()) {
        throw std::runtime_error("Residual weight data exceeds buffer");
      }

      config.parameters.residualParams[i].weights = readFloatVector(blockPtr + BLOCK_HEADER_SIZE, dataSize);
      pos += BLOCK_HEADER_SIZE + dataSize;

      // RESIDUAL_BIASES
      if (pos + BLOCK_HEADER_SIZE > data.size()) {
        throw std::runtime_error("Unexpected end of binary parameter data");
      }

      blockPtr = data.data() + pos;
      blockType = static_cast<uint8_t>(blockPtr[0]);
      dataSize = readU32LE(blockPtr + 18);

      if (blockType != BLOCK_RESIDUAL_BIASES) {
        throw std::runtime_error("Expected RESIDUAL_BIASES block");
      }

      if (pos + BLOCK_HEADER_SIZE + dataSize > data.size()) {
        throw std::runtime_error("Residual bias data exceeds buffer");
      }

      config.parameters.residualParams[i].biases = readFloatVector(blockPtr + BLOCK_HEADER_SIZE, dataSize);
      pos += BLOCK_HEADER_SIZE + dataSize;

      // Set residual metadata from precomputed shape
      config.parameters.residualParams[i].inC = residualMetaVec[i].inC;
      config.parameters.residualParams[i].outC = residualMetaVec[i].outC;
    }

    //-- Read dense parameters --//

    // The ANN core stores parameters with an entry at index 0 for the input layer
    // (an empty placeholder — the input layer has no weights). So the binary contains
    // denseLayers.size() + 1 weight/bias pairs. The serializer writes
    // denseParams.weights.size() entries, which equals numLayers (input + dense).
    size_t numDenseParamEntries = layersConfig.denseLayers.size() + 1;
    config.parameters.denseParams.weights.resize(numDenseParamEntries);
    config.parameters.denseParams.biases.resize(numDenseParamEntries);

    for (size_t i = 0; i < numDenseParamEntries; ++i) {
      // Dense weights (ANN_WEIGHTS block)
      if (pos + BLOCK_HEADER_SIZE > data.size()) {
        throw std::runtime_error("Unexpected end of binary parameter data");
      }

      const char* blockPtr = data.data() + pos;
      uint8_t blockType = static_cast<uint8_t>(blockPtr[0]);
      uint32_t dim0 = readU32LE(blockPtr + 6);
      uint32_t dim1 = readU32LE(blockPtr + 10);
      uint32_t dataSize = readU32LE(blockPtr + 18);

      if (blockType != BLOCK_ANN_WEIGHTS) {
        throw std::runtime_error("Expected ANN_WEIGHTS block for dense layer");
      }

      if (pos + BLOCK_HEADER_SIZE + dataSize > data.size()) {
        throw std::runtime_error("Dense weight data exceeds buffer");
      }

      std::vector<float> flatWeights = readFloatVector(blockPtr + BLOCK_HEADER_SIZE, dataSize);

      config.parameters.denseParams.weights[i].resize(dim0);

      size_t flatIdx = 0;

      for (uint32_t n = 0; n < dim0; ++n) {
        config.parameters.denseParams.weights[i][n].resize(dim1);

        for (uint32_t w = 0; w < dim1; ++w) {
          config.parameters.denseParams.weights[i][n][w] = flatWeights[flatIdx++];
        }
      }

      pos += BLOCK_HEADER_SIZE + dataSize;

      // Dense biases (ANN_BIASES block)
      if (pos + BLOCK_HEADER_SIZE > data.size()) {
        throw std::runtime_error("Unexpected end of binary parameter data");
      }

      blockPtr = data.data() + pos;
      blockType = static_cast<uint8_t>(blockPtr[0]);
      dataSize = readU32LE(blockPtr + 18);

      if (blockType != BLOCK_ANN_BIASES) {
        throw std::runtime_error("Expected ANN_BIASES block for dense layer");
      }

      if (pos + BLOCK_HEADER_SIZE + dataSize > data.size()) {
        throw std::runtime_error("Dense bias data exceeds buffer");
      }

      config.parameters.denseParams.biases[i] = readFloatVector(blockPtr + BLOCK_HEADER_SIZE, dataSize);
      pos += BLOCK_HEADER_SIZE + dataSize;
    }
  }

  //===================================================================================================================//
  //-- buildCNNModelJson --//
  //===================================================================================================================//

  nlohmann::ordered_json ModelSerializer::buildCNNModelJson(const CNN::Core<float>& core,
                                                            const CNN::CoreConfig<float>& coreConfig,
                                                            const IOConfig& ioConfig,
                                                            const AugmentationConfig& augConfig,
                                                            const ValidationMetadata& validationMeta)
  {
    nlohmann::ordered_json json;

    json["mode"] = Common::Mode::typeToName(core.getModeType());
    json["device"] = Common::Device::typeToName(core.getDeviceType());
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
    serializeTrainConfig(tcJson, core.getTrainConfig());
    serializeAugConfig(tcJson, augConfig);
    serializeValidationConfig(tcJson, augConfig);
    serializeMonitoringConfig(tcJson, core);
    json["train"] = tcJson;

    // Test config
    nlohmann::ordered_json testJson;
    serializeTestConfig(testJson, coreConfig.testConfig);
    json["test"] = testJson;

    // Training metadata
    const auto& md = core.getTrainMetadata();
    nlohmann::ordered_json mdJson;
    serializeTrainMetadata(mdJson, md);
    serializeValidationMeta(mdJson, validationMeta);
    json["trainMetadata"] = mdJson;

    return json;
  }

  //===================================================================================================================//
  //-- saveCNNModelToPackage --//
  //===================================================================================================================//

  void ModelSerializer::saveCNNModelToPackage(const std::string& packagePath, const CNN::Core<float>& core,
                                              const CNN::CoreConfig<float>& coreConfig, const IOConfig& ioConfig,
                                              const AugmentationConfig& augConfig,
                                              const ValidationMetadata& validationMeta)
  {
    auto json = buildCNNModelJson(core, coreConfig, ioConfig, augConfig, validationMeta);
    auto binData = serializeCNNParametersBinary(core);
    auto jsonStr = json.dump(4);
    ModelPackage::createFromMemory(packagePath, jsonStr, binData);
  }

} // namespace NN_CLI
