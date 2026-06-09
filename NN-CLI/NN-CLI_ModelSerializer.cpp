#include "NN-CLI_ModelSerializer.hpp"

#include "NN-CLI_DataType.hpp"
#include "NN-CLI_ModelPackage.hpp"
#include "NN-CLI_Utils.hpp"

#include <CNN_SlidingStrategy.hpp>

#include <QDir>
#include <QFile>
#include <QFileInfo>

#include <json.hpp>

#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stack>
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
  //-- Helper: binary format constants --//
  //===================================================================================================================//

  static constexpr uint32_t BINARY_MAGIC = 0xAE10AE01;
  static constexpr uint16_t BINARY_HEADER_SIZE = 16;
  static constexpr uint8_t BINARY_VERSION = 1;
  static constexpr uint8_t BINARY_ENDIANNESS_LE = 0;
  static constexpr uint8_t BINARY_MODEL_ANN = 0;
  static constexpr uint8_t BINARY_MODEL_CNN = 1;

  enum BinaryBlockType : uint8_t {
    BLOCK_ANN_WEIGHTS = 0,
    BLOCK_ANN_BIASES = 1,
    BLOCK_CONV_FILTERS = 2,
    BLOCK_CONV_BIASES = 3,
    BLOCK_NORM_GAMMA = 4,
    BLOCK_NORM_BETA = 5,
    BLOCK_NORM_RUNNING_MEAN = 6,
    BLOCK_NORM_RUNNING_VAR = 7,
    BLOCK_RESIDUAL_WEIGHTS = 8,
    BLOCK_RESIDUAL_BIASES = 9
  };

  static constexpr size_t BLOCK_HEADER_SIZE = 22;

  //===================================================================================================================//
  //-- Helper: write binary header to buffer --//
  //===================================================================================================================//

  static void writeBinaryHeader(std::vector<char>& buffer, uint8_t modelType)
  {
    size_t offset = buffer.size();
    buffer.resize(offset + BINARY_HEADER_SIZE);

    char* ptr = buffer.data() + offset;
    std::memset(ptr, 0, BINARY_HEADER_SIZE);

    uint32_t magic = BINARY_MAGIC;
    uint16_t headerSize = BINARY_HEADER_SIZE;
    uint8_t version = BINARY_VERSION;
    uint8_t endianness = BINARY_ENDIANNESS_LE;

    std::memcpy(ptr + 0, &magic, 4);
    std::memcpy(ptr + 4, &headerSize, 2);
    std::memcpy(ptr + 6, &version, 1);
    std::memcpy(ptr + 7, &endianness, 1);
    std::memcpy(ptr + 8, &modelType, 1);
    // bytes 9-15 reserved (zeroed above)
  }

  //===================================================================================================================//
  //-- Helper: write binary data block to buffer --//
  //===================================================================================================================//

  static void writeBlockToBuffer(std::vector<char>& buffer, uint8_t blockType, uint32_t layerIdx,
                                 uint8_t ndim, uint32_t dim0, uint32_t dim1, uint32_t dim2,
                                 const std::vector<float>& data)
  {
    uint32_t dataSize = static_cast<uint32_t>(data.size() * sizeof(float));

    size_t offset = buffer.size();
    buffer.resize(offset + BLOCK_HEADER_SIZE + dataSize);

    char* ptr = buffer.data() + offset;

    std::memcpy(ptr + 0, &blockType, 1);
    std::memcpy(ptr + 1, &layerIdx, 4);
    std::memcpy(ptr + 5, &ndim, 1);
    std::memcpy(ptr + 6, &dim0, 4);
    std::memcpy(ptr + 10, &dim1, 4);
    std::memcpy(ptr + 14, &dim2, 4);
    std::memcpy(ptr + 18, &dataSize, 4);

    if (dataSize > 0) {
      std::memcpy(ptr + BLOCK_HEADER_SIZE, data.data(), dataSize);
    }
  }

  //===================================================================================================================//
  //-- Helper: flatten 2D weights to 1D float vector --//
  //===================================================================================================================//

  static std::vector<float> flattenWeights(const ANN::Tensor2D<float>& weightMatrix)
  {
    size_t totalSize = 0;

    for (const auto& row : weightMatrix) {
      totalSize += row.size();
    }

    std::vector<float> flat;
    flat.reserve(totalSize);

    for (const auto& row : weightMatrix) {
      flat.insert(flat.end(), row.begin(), row.end());
    }

    return flat;
  }

  //===================================================================================================================//
  //-- Helper: serialize ANN parameters to binary buffer --//
  //===================================================================================================================//

  static std::vector<char> serializeANNParametersBinary(const ANN::Core<float>& core)
  {
    std::vector<char> buffer;
    writeBinaryHeader(buffer, BINARY_MODEL_ANN);

    const auto& params = core.getParameters();

    for (size_t i = 0; i < params.weights.size(); ++i) {
      uint32_t layerIdx = static_cast<uint32_t>(i);

      // Weights: ndim=2, dim0=numNeurons, dim1=numWeightsPerNeuron
      const auto& weightMatrix = params.weights[i];
      uint32_t dim0 = static_cast<uint32_t>(weightMatrix.size());
      uint32_t dim1 = (dim0 > 0) ? static_cast<uint32_t>(weightMatrix[0].size()) : 0u;

      std::vector<float> flatWeights = flattenWeights(weightMatrix);
      writeBlockToBuffer(buffer, BLOCK_ANN_WEIGHTS, layerIdx, 2, dim0, dim1, 0, flatWeights);

      // Biases: ndim=1, dim0=numBiases
      const auto& biasVec = params.biases[i];
      uint32_t numBiases = static_cast<uint32_t>(biasVec.size());
      std::vector<float> biasData(biasVec.begin(), biasVec.end());
      writeBlockToBuffer(buffer, BLOCK_ANN_BIASES, layerIdx, 1, numBiases, 0, 0, biasData);
    }

    return buffer;
  }

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

      writeBlockToBuffer(buffer, BLOCK_CONV_FILTERS, layerIdx, 1,
                         static_cast<uint32_t>(cp.filters.size()), 0, 0, cp.filters);

      writeBlockToBuffer(buffer, BLOCK_CONV_BIASES, layerIdx, 1,
                         static_cast<uint32_t>(cp.biases.size()), 0, 0, cp.biases);
    }

    // Norm parameters: GAMMA + BETA + RUNNING_MEAN + RUNNING_VAR per layer
    for (size_t i = 0; i < params.normParams.size(); ++i) {
      uint32_t layerIdx = static_cast<uint32_t>(i);
      const auto& np = params.normParams[i];

      writeBlockToBuffer(buffer, BLOCK_NORM_GAMMA, layerIdx, 1,
                         static_cast<uint32_t>(np.gamma.size()), 0, 0, np.gamma);
      writeBlockToBuffer(buffer, BLOCK_NORM_BETA, layerIdx, 1,
                         static_cast<uint32_t>(np.beta.size()), 0, 0, np.beta);
      writeBlockToBuffer(buffer, BLOCK_NORM_RUNNING_MEAN, layerIdx, 1,
                         static_cast<uint32_t>(np.runningMean.size()), 0, 0, np.runningMean);
      writeBlockToBuffer(buffer, BLOCK_NORM_RUNNING_VAR, layerIdx, 1,
                         static_cast<uint32_t>(np.runningVar.size()), 0, 0, np.runningVar);
    }

    // Residual parameters: WEIGHTS + BIASES per layer
    for (size_t i = 0; i < params.residualParams.size(); ++i) {
      uint32_t layerIdx = static_cast<uint32_t>(i);
      const auto& rp = params.residualParams[i];

      writeBlockToBuffer(buffer, BLOCK_RESIDUAL_WEIGHTS, layerIdx, 1,
                         static_cast<uint32_t>(rp.weights.size()), 0, 0, rp.weights);
      writeBlockToBuffer(buffer, BLOCK_RESIDUAL_BIASES, layerIdx, 1,
                         static_cast<uint32_t>(rp.biases.size()), 0, 0, rp.biases);
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
  //-- Helper: read little-endian uint32 from byte buffer --//
  //===================================================================================================================//

  static uint32_t readU32LE(const char* ptr)
  {
    uint32_t val;
    std::memcpy(&val, ptr, 4);
    return val;
  }

  //===================================================================================================================//
  //-- Helper: read little-endian uint16 from byte buffer --//
  //===================================================================================================================//

  static uint16_t readU16LE(const char* ptr)
  {
    uint16_t val;
    std::memcpy(&val, ptr, 2);
    return val;
  }

  //===================================================================================================================//
  //-- Helper: read float vector from byte buffer --//
  //===================================================================================================================//

  static std::vector<float> readFloatVector(const char* ptr, uint32_t dataSize)
  {
    size_t numFloats = dataSize / sizeof(float);
    std::vector<float> result(numFloats);

    if (numFloats > 0) {
      std::memcpy(result.data(), ptr, dataSize);
    }

    return result;
  }

  //===================================================================================================================//
  //-- saveANNParametersBinary --//
  //===================================================================================================================//

  void ModelSerializer::saveANNParametersBinary(const std::string& binPath, const ANN::Core<float>& core)
  {
    std::vector<char> buffer = serializeANNParametersBinary(core);

    std::ofstream ofs(binPath, std::ios::binary);

    if (!ofs) {
      throw std::runtime_error("Failed to open binary parameter file for writing: " + binPath);
    }

    ofs.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    ofs.close();
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
  //-- loadANNParametersBinary --//
  //===================================================================================================================//

  void ModelSerializer::loadANNParametersBinary(const std::vector<char>& data,
                                             ANN::CoreConfig<float>& config,
                                             const ANN::LayersConfig& layersConfig)
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

    // Validate model type (ANN)
    uint8_t modelType = static_cast<uint8_t>(ptr[8]);

    if (modelType != BINARY_MODEL_ANN) {
      throw std::runtime_error("Binary parameter data is not an ANN model");
    }

    size_t pos = BINARY_HEADER_SIZE;

    config.parameters.weights.resize(layersConfig.size());
    config.parameters.biases.resize(layersConfig.size());

    for (size_t layerIdx = 0; layerIdx < layersConfig.size(); ++layerIdx) {
      // Read weights block
      if (pos + BLOCK_HEADER_SIZE > data.size()) {
        throw std::runtime_error("Unexpected end of binary parameter data");
      }

      const char* blockPtr = data.data() + pos;
      uint8_t blockType = static_cast<uint8_t>(blockPtr[0]);
      uint32_t blockIdx = readU32LE(blockPtr + 1);
      uint8_t ndim = static_cast<uint8_t>(blockPtr[5]);
      uint32_t dim0 = readU32LE(blockPtr + 6);
      uint32_t dim1 = readU32LE(blockPtr + 10);
      uint32_t dataSize = readU32LE(blockPtr + 18);

      if (blockType != BLOCK_ANN_WEIGHTS) {
        throw std::runtime_error("Expected ANN_WEIGHTS block at layer " + std::to_string(layerIdx));
      }

      if (blockIdx != layerIdx) {
        throw std::runtime_error("Weight block layer index mismatch");
      }

      if (pos + BLOCK_HEADER_SIZE + dataSize > data.size()) {
        throw std::runtime_error("Weight data exceeds buffer");
      }

      // Reshape flat data into Tensor2D (vector of vectors)
      std::vector<float> flatWeights = readFloatVector(blockPtr + BLOCK_HEADER_SIZE, dataSize);

      config.parameters.weights[layerIdx].resize(dim0);

      size_t flatIdx = 0;

      for (uint32_t n = 0; n < dim0; ++n) {
        config.parameters.weights[layerIdx][n].resize(dim1);

        for (uint32_t w = 0; w < dim1; ++w) {
          config.parameters.weights[layerIdx][n][w] = flatWeights[flatIdx++];
        }
      }

      pos += BLOCK_HEADER_SIZE + dataSize;

      // Read biases block
      if (pos + BLOCK_HEADER_SIZE > data.size()) {
        throw std::runtime_error("Unexpected end of binary parameter data");
      }

      blockPtr = data.data() + pos;
      blockType = static_cast<uint8_t>(blockPtr[0]);
      blockIdx = readU32LE(blockPtr + 1);
      dataSize = readU32LE(blockPtr + 18);

      if (blockType != BLOCK_ANN_BIASES) {
        throw std::runtime_error("Expected ANN_BIASES block at layer " + std::to_string(layerIdx));
      }

      if (blockIdx != layerIdx) {
        throw std::runtime_error("Bias block layer index mismatch");
      }

      if (pos + BLOCK_HEADER_SIZE + dataSize > data.size()) {
        throw std::runtime_error("Bias data exceeds buffer");
      }

      config.parameters.biases[layerIdx] = readFloatVector(blockPtr + BLOCK_HEADER_SIZE, dataSize);

      pos += BLOCK_HEADER_SIZE + dataSize;
    }
  }

  //===================================================================================================================//
  //-- loadCNNParametersBinary --//
  //===================================================================================================================//

  void ModelSerializer::loadCNNParametersBinary(const std::vector<char>& data,
                                                 CNN::CoreConfig<float>& config,
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
    struct ConvMeta { ulong numFilters; ulong inputC; ulong filterH; ulong filterW; };
    struct NormMeta { ulong numChannels; };
    struct ResidualMeta { ulong inC; ulong outC; };

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

    size_t numDenseLayers = layersConfig.denseLayers.size();
    config.parameters.denseParams.weights.resize(numDenseLayers);
    config.parameters.denseParams.biases.resize(numDenseLayers);

    for (size_t i = 0; i < numDenseLayers; ++i) {
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
  //-- buildANNModelJson --//
  //===================================================================================================================//

  nlohmann::ordered_json ModelSerializer::buildANNModelJson(const ANN::Core<float>& core,
                                                            const ANN::CoreConfig<float>& coreConfig,
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

    return json;
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

    return json;
  }

  //===================================================================================================================//
  //-- saveANNModelToPackage --//
  //===================================================================================================================//

  void ModelSerializer::saveANNModelToPackage(const std::string& packagePath,
                                           const ANN::Core<float>& core,
                                           const ANN::CoreConfig<float>& coreConfig,
                                           const IOConfig& ioConfig,
                                           const AugmentationConfig& augConfig,
                                           const ValidationMetadata& validationMeta)
  {
    auto json = buildANNModelJson(core, coreConfig, ioConfig, augConfig, validationMeta);
    auto binData = serializeANNParametersBinary(core);
    auto jsonStr = json.dump(4);
    ModelPackage::createFromMemory(packagePath, jsonStr, binData);
  }

  //===================================================================================================================//
  //-- saveCNNModelToPackage --//
  //===================================================================================================================//

  void ModelSerializer::saveCNNModelToPackage(const std::string& packagePath,
                                              const CNN::Core<float>& core,
                                              const CNN::CoreConfig<float>& coreConfig,
                                              const IOConfig& ioConfig,
                                              const AugmentationConfig& augConfig,
                                              const ValidationMetadata& validationMeta)
  {
    auto json = buildCNNModelJson(core, coreConfig, ioConfig, augConfig, validationMeta);
    auto binData = serializeCNNParametersBinary(core);
    auto jsonStr = json.dump(4);
    ModelPackage::createFromMemory(packagePath, jsonStr, binData);
  }

  //===================================================================================================================//
  //-- Output path helpers --//
  //===================================================================================================================//

  std::string ModelSerializer::generateTrainingFilename(ulong epochs, ulong samples, float loss)
  {
    std::ostringstream oss;
    oss << "trained_E-" << epochs << "_S-" << samples << "_L-" << std::fixed << std::setprecision(6) << loss
        << ".nnmodel.tar";
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
    oss << "checkpoint_E-" << epoch << "_L-" << std::fixed << std::setprecision(6) << loss << ".nnmodel.tar";

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

    QString outputPath = outputDir.filePath("best_model.nnmodel.tar");
    return outputPath.toStdString();
  }

} // namespace NN_CLI
