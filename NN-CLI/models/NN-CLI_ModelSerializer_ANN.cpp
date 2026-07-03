#include "NN-CLI_ModelSerializer.hpp"
#include "NN-CLI_ModelSerializerDetail.hpp"

#include <cstring>
#include <fstream>

#include "NN-CLI_DataType.hpp"
#include "NN-CLI_ModelPackage.hpp"

namespace NN_CLI
{

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
  //-- loadANNParametersBinary --//
  //===================================================================================================================//

  void ModelSerializer::loadANNParametersBinary(const std::vector<char>& data, ANN::CoreConfig<float>& config,
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
  //-- saveANNModelToPackage --//
  //===================================================================================================================//

  void ModelSerializer::saveANNModelToPackage(const std::string& packagePath, const ANN::Core<float>& core,
                                              const ANN::CoreConfig<float>& coreConfig, const IOConfig& ioConfig,
                                              const AugmentationConfig& augConfig,
                                              const ValidationMetadata& validationMeta)
  {
    auto json = buildANNModelJson(core, coreConfig, ioConfig, augConfig, validationMeta);
    auto binData = serializeANNParametersBinary(core);
    auto jsonStr = json.dump(4);
    ModelPackage::createFromMemory(packagePath, jsonStr, binData);
  }

} // namespace NN_CLI
