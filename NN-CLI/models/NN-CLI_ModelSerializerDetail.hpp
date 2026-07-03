#ifndef NN_CLI_MODELSERIALIZERDETAIL_HPP
#define NN_CLI_MODELSERIALIZERDETAIL_HPP

#include <cstdint>
#include <string>
#include <vector>

#include <json.hpp>

#include "NN-CLI_ModelSerializer.hpp"

namespace NN_CLI
{

  //===================================================================================================================//
  //-- Binary format constants (shared across serializer TUs) --//
  //===================================================================================================================//
  constexpr uint32_t BINARY_MAGIC = 0xAE10AE01;
  constexpr uint16_t BINARY_HEADER_SIZE = 16;
  constexpr uint8_t BINARY_VERSION = 1;
  constexpr uint8_t BINARY_ENDIANNESS_LE = 0;
  constexpr uint8_t BINARY_MODEL_ANN = 0;
  constexpr uint8_t BINARY_MODEL_CNN = 1;

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

  constexpr size_t BLOCK_HEADER_SIZE = 22;

  //===================================================================================================================//
  //-- Binary primitive declarations --//
  //===================================================================================================================//

  void writeBinaryHeader(std::vector<char>& buffer, uint8_t modelType);

  void writeBlockToBuffer(std::vector<char>& buffer, uint8_t blockType, uint32_t layerIdx, uint8_t ndim, uint32_t dim0,
                          uint32_t dim1, uint32_t dim2, const std::vector<float>& data);

  std::vector<float> flattenWeights(const ANN::Tensor2D<float>& weightMatrix);

  uint32_t readU32LE(const char* ptr);

  uint16_t readU16LE(const char* ptr);

  std::vector<float> readFloatVector(const char* ptr, uint32_t dataSize);

  //===================================================================================================================//
  //-- Non-template JSON helper declarations --//
  //===================================================================================================================//

  void serializeAugConfig(nlohmann::ordered_json& tcJson, const AugmentationConfig& augConfig);

  void serializeValidationConfig(nlohmann::ordered_json& tcJson, const AugmentationConfig& augConfig);

  void serializeValidationMeta(nlohmann::ordered_json& mdJson, const ValidationMetadata& validationMeta);

  //===================================================================================================================//
  //-- Template JSON helper definitions (header-inline, shared across TUs) --//
  //===================================================================================================================//

  template <typename CoreT>
  void serializeMonitoringConfig(nlohmann::ordered_json& tcJson, const CoreT& core)
  {
    const auto& mc = core.getTrainConfig().monitoringConfig;
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

  template <typename TrainConfigT>
  void serializeTrainConfig(nlohmann::ordered_json& tcJson, const TrainConfigT& tc)
  {
    tcJson["numEpochs"] = tc.numEpochs;
    tcJson["learningRate"] = tc.learningRate;
    tcJson["batchSize"] = tc.batchSize;
    tcJson["fetchSize"] = tc.fetchSize;
    tcJson["shuffleSamples"] = tc.shuffleSamples;

    tcJson["dropoutRate"] = tc.dropoutRate;

    nlohmann::ordered_json optJson;
    optJson["type"] = Common::optimizerTypeToName(tc.optimizer.type);
    optJson["beta1"] = tc.optimizer.beta1;
    optJson["beta2"] = tc.optimizer.beta2;
    optJson["epsilon"] = tc.optimizer.epsilon;
    tcJson["optimizer"] = optJson;

    using SchedulerType = std::decay_t<decltype(tc.learningRateScheduler.type)>;

    if (tc.learningRateScheduler.type != SchedulerType::NONE) {
      nlohmann::ordered_json schedJson;
      using SchedulerConfigT = std::decay_t<decltype(tc.learningRateScheduler)>;
      schedJson["type"] = SchedulerConfigT::typeToName(tc.learningRateScheduler.type);
      schedJson["gamma"] = tc.learningRateScheduler.gamma;
      schedJson["stepSize"] = tc.learningRateScheduler.stepSize;
      schedJson["minLearningRate"] = tc.learningRateScheduler.minLearningRate;
      schedJson["patience"] = tc.learningRateScheduler.patience;
      schedJson["minDelta"] = tc.learningRateScheduler.minDelta;
      tcJson["learningRateScheduler"] = schedJson;
    }
  }

  template <typename TestConfigT>
  void serializeTestConfig(nlohmann::ordered_json& testJson, const TestConfigT& testConfig)
  {
    testJson["fetchSize"] = testConfig.fetchSize;
  }

  template <typename MetadataT>
  void serializeTrainMetadata(nlohmann::ordered_json& mdJson, const MetadataT& md)
  {
    mdJson["startTime"] = md.startTime;
    mdJson["endTime"] = md.endTime;
    mdJson["durationSeconds"] = md.durationSeconds;
    mdJson["durationFormatted"] = md.durationFormatted;
    mdJson["numSamples"] = md.numSamples;
    mdJson["finalLoss"] = md.finalLoss;

    // Serialize the 0-based index of the last completed epoch whenever any
    // epoch ran. Sourced from epochHistory so it always matches the serialized
    // epochs[] (a single-epoch run has lastEpoch == 0, which an "> 0" guard
    // would wrongly drop).
    if (!md.epochHistory.empty())
      mdJson["lastEpoch"] = md.epochHistory.back().epoch;

    if (!md.stopReason.empty())
      mdJson["stopReason"] = md.stopReason;

    if (md.bestEpoch > 0) {
      mdJson["bestEpoch"] = md.bestEpoch;
      mdJson["bestLoss"] = md.bestLoss;
    }

    // Persist learning-rate scheduler state so training can resume (matters for plateau, which tracks
    // bestValidationLoss and a patience counter across epochs).
    if (md.learningRateSchedulerState.initialized) {
      nlohmann::ordered_json ssJson;
      ssJson["currentLearningRate"] = md.learningRateSchedulerState.currentLearningRate;
      ssJson["baseLearningRate"] = md.learningRateSchedulerState.baseLearningRate;
      ssJson["epochsSinceImprovement"] = md.learningRateSchedulerState.epochsSinceImprovement;
      ssJson["bestValidationLoss"] = md.learningRateSchedulerState.bestValidationLoss;
      ssJson["initialized"] = md.learningRateSchedulerState.initialized;
      mdJson["learningRateSchedulerState"] = ssJson;
    }

    // Serialize epoch history (only when non-empty)
    if (!md.epochHistory.empty()) {
      nlohmann::ordered_json epochsArr = nlohmann::ordered_json::array();

      for (const auto& record : md.epochHistory) {
        nlohmann::ordered_json recordJson;
        recordJson["epoch"] = record.epoch;
        recordJson["loss"] = static_cast<double>(record.loss);
        recordJson["learningRate"] = static_cast<double>(record.learningRate);

        if (record.hasValLoss) {
          recordJson["valLoss"] = static_cast<double>(record.valLoss);
          recordJson["hasValLoss"] = true;
        } else {
          recordJson["hasValLoss"] = false;
        }

        recordJson["isBest"] = record.isBest;
        recordJson["completionTime"] = record.completionTime;

        if (record.hasValConfusionMatrix && !record.valConfusionMatrix.empty()) {
          ModelSerializer::serializeConfusionMatrix(recordJson, "valConfusionMatrix", record.valConfusionMatrix);
        }

        epochsArr.push_back(recordJson);
      }

      mdJson["epochs"] = epochsArr;
    }
  }

} // namespace NN_CLI

#endif // NN_CLI_MODELSERIALIZERDETAIL_HPP
