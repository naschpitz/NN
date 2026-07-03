#include "NN-CLI_ModelSerializerDetail.hpp"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <QDir>
#include <QFile>
#include <QFileInfo>

#include "NN-CLI_Utils.hpp"

namespace NN_CLI
{

  //===================================================================================================================//
  //-- Helper: serialize augmentation config into training config JSON --//
  //===================================================================================================================//

  void serializeAugConfig(nlohmann::ordered_json& tcJson, const AugmentationConfig& augConfig)
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

  void serializeValidationConfig(nlohmann::ordered_json& tcJson, const AugmentationConfig& augConfig)
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
  //-- Helper: serialize validation metadata --//
  //===================================================================================================================//

  void serializeValidationMeta(nlohmann::ordered_json& mdJson, const ValidationMetadata& validationMeta)
  {
    if (validationMeta.enabled) {
      mdJson["numValidationSamples"] = validationMeta.numValSamples;
      mdJson["finalValidationLoss"] = validationMeta.lastValLoss;
      mdJson["bestValidationLoss"] = validationMeta.bestValidationLoss;
      mdJson["bestValidationEpoch"] = validationMeta.bestValEpoch;
    }
  }

  //===================================================================================================================//
  //-- Helper: serialize a confusion matrix under parent[key] --//
  //===================================================================================================================//

  template <typename T>
  void ModelSerializer::serializeConfusionMatrix(nlohmann::ordered_json& parent, const std::string& key,
                                                 const Common::ConfusionMatrix<T>& cm)
  {
    nlohmann::ordered_json cmJson;
    cmJson["numClasses"] = cm.numClasses;
    cmJson["totalSamples"] = cm.totalSamples;

    nlohmann::ordered_json matrixArr = nlohmann::ordered_json::array();

    for (size_t i = 0; i < cm.matrix.size(); i++) {
      matrixArr.push_back(cm.matrix[i]);
    }

    cmJson["matrix"] = matrixArr;
    cmJson["accuracy"] = static_cast<double>(cm.accuracy);

    nlohmann::ordered_json perClassArr = nlohmann::ordered_json::array();

    for (size_t c = 0; c < cm.numClasses; c++) {
      nlohmann::ordered_json pc;
      pc["tp"] = cm.truePositive[c];
      pc["fp"] = cm.falsePositive[c];
      pc["fn"] = cm.falseNegative[c];
      pc["tn"] = cm.trueNegative[c];
      pc["precision"] = static_cast<double>(cm.precision[c]);
      pc["recall"] = static_cast<double>(cm.recall[c]);
      pc["f1"] = static_cast<double>(cm.f1Score[c]);
      pc["support"] = cm.support[c];
      perClassArr.push_back(pc);
    }

    cmJson["perClass"] = perClassArr;

    nlohmann::ordered_json macroJson;
    macroJson["precision"] = static_cast<double>(cm.macroPrecision);
    macroJson["recall"] = static_cast<double>(cm.macroRecall);
    macroJson["f1"] = static_cast<double>(cm.macroF1);
    cmJson["macro"] = macroJson;

    nlohmann::ordered_json microJson;
    microJson["precision"] = static_cast<double>(cm.microPrecision);
    microJson["recall"] = static_cast<double>(cm.microRecall);
    microJson["f1"] = static_cast<double>(cm.microF1);
    cmJson["micro"] = microJson;

    nlohmann::ordered_json weightedJson;
    weightedJson["precision"] = static_cast<double>(cm.weightedPrecision);
    weightedJson["recall"] = static_cast<double>(cm.weightedRecall);
    weightedJson["f1"] = static_cast<double>(cm.weightedF1);
    cmJson["weighted"] = weightedJson;

    parent[key] = cmJson;
  }

  //===================================================================================================================//
  //-- Helper: write binary header to buffer --//
  //===================================================================================================================//

  void writeBinaryHeader(std::vector<char>& buffer, uint8_t modelType)
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

  void writeBlockToBuffer(std::vector<char>& buffer, uint8_t blockType, uint32_t layerIdx, uint8_t ndim, uint32_t dim0,
                          uint32_t dim1, uint32_t dim2, const std::vector<float>& data)
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

  std::vector<float> flattenWeights(const ANN::Tensor2D<float>& weightMatrix)
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
  //-- Helper: read little-endian uint32 from byte buffer --//
  //===================================================================================================================//

  uint32_t readU32LE(const char* ptr)
  {
    uint32_t val;
    std::memcpy(&val, ptr, 4);
    return val;
  }

  //===================================================================================================================//
  //-- Helper: read little-endian uint16 from byte buffer --//
  //===================================================================================================================//

  uint16_t readU16LE(const char* ptr)
  {
    uint16_t val;
    std::memcpy(&val, ptr, 2);
    return val;
  }

  //===================================================================================================================//
  //-- Helper: read float vector from byte buffer --//
  //===================================================================================================================//

  std::vector<float> readFloatVector(const char* ptr, uint32_t dataSize)
  {
    size_t numFloats = dataSize / sizeof(float);
    std::vector<float> result(numFloats);

    if (numFloats > 0) {
      std::memcpy(result.data(), ptr, dataSize);
    }

    return result;
  }

  //===================================================================================================================//
  //-- Output path helpers --//
  //===================================================================================================================//

  std::string ModelSerializer::generateTrainFilename(ulong epochs, ulong samples, float loss)
  {
    std::ostringstream oss;
    oss << "trained_E-" << epochs << "_S-" << samples << "_L-" << std::fixed << std::setprecision(6) << loss
        << ".nnmodel.tar";
    return oss.str();
  }

  //===================================================================================================================//

  std::string ModelSerializer::generateDefaultOutputPath(const QString& outputDir, ulong epochs, ulong samples,
                                                         float loss)
  {
    QDir dir(outputDir);
    QString outputPath = dir.filePath(QString::fromStdString(generateTrainFilename(epochs, samples, loss)));
    return outputPath.toStdString();
  }

  //===================================================================================================================//

  std::string ModelSerializer::generateCheckpointPath(const QString& outputDir, ulong epoch, float loss)
  {
    QDir dir(outputDir);

    std::ostringstream oss;
    oss << "checkpoint_E-" << epoch << "_L-" << std::fixed << std::setprecision(6) << loss << ".nnmodel.tar";

    QString outputPath = dir.filePath(QString::fromStdString(oss.str()));
    return outputPath.toStdString();
  }

  //===================================================================================================================//

  std::string ModelSerializer::generateBestModelPath(const QString& outputDir)
  {
    QDir dir(outputDir);
    QString outputPath = dir.filePath("best_model.nnmodel.tar");
    return outputPath.toStdString();
  }

  //===================================================================================================================//
  //-- Explicit template instantiations --//
  //===================================================================================================================//

  template void ModelSerializer::serializeConfusionMatrix<float>(nlohmann::ordered_json& parent, const std::string& key,
                                                                 const Common::ConfusionMatrix<float>& cm);

} // namespace NN_CLI
