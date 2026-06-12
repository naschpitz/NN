#include "NN-CLI_Runner.hpp"

#include "ANN/ANN_Core.hpp"
#include "CNN/CNN_Core.hpp"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <string>

//===================================================================================================================//
//  Template implementations
//===================================================================================================================//

template <typename CoreT, typename CoreConfigT>
NN_CLI::Runner<CoreT, CoreConfigT>::Runner(const QCommandLineParser& parser, NN_CLI::LogLevel logLevel,
                                           NN_CLI::IOConfig& ioConfig, NN_CLI::AugmentationConfig& augConfig,
                                           std::unique_ptr<CoreT>& core, CoreConfigT& coreConfig)
  : parser(parser),
    logLevel(logLevel),
    ioConfig(ioConfig),
    augConfig(augConfig),
    core(core),
    coreConfig(coreConfig)
{
}

//===================================================================================================================//
//  Observer management
//===================================================================================================================//

template <typename CoreT, typename CoreConfigT>
void NN_CLI::Runner<CoreT, CoreConfigT>::addObserver(NN_CLI::IRunnerObserver* observer)
{
  if (observer == nullptr)
    return;

  // Avoid duplicates.
  for (auto* existing : this->observers) {
    if (existing == observer)
      return;
  }

  this->observers.push_back(observer);
}

//===================================================================================================================//

template <typename CoreT, typename CoreConfigT>
void NN_CLI::Runner<CoreT, CoreConfigT>::removeObserver(NN_CLI::IRunnerObserver* observer)
{
  if (observer == nullptr)
    return;

  auto it = std::find(this->observers.begin(), this->observers.end(), observer);

  if (it != this->observers.end())
    this->observers.erase(it);
}

//===================================================================================================================//
//  Observer notifications
//===================================================================================================================//

template <typename CoreT, typename CoreConfigT>
void NN_CLI::Runner<CoreT, CoreConfigT>::notifySampleLoadProgress(ulong current, ulong total, ulong batchIndex,
                                                                  ulong totalBatches, bool isValidation)
{
  for (auto* observer : this->observers)
    observer->onSampleLoadProgress(current, total, batchIndex, totalBatches, isValidation);
}

//===================================================================================================================//

template <typename CoreT, typename CoreConfigT>
void NN_CLI::Runner<CoreT, CoreConfigT>::notifyValidationProgress(ulong current, ulong total)
{
  for (auto* observer : this->observers)
    observer->onValidationProgress(current, total);
}

//===================================================================================================================//

template <typename CoreT, typename CoreConfigT>
void NN_CLI::Runner<CoreT, CoreConfigT>::notifyBatchProgress(int batchIdx, int totalBatches, float currentLoss,
                                                              float samplesPerSec, float etaSeconds,
                                                              const std::vector<float>& fractions)
{
  for (auto* observer : this->observers)
    observer->onBatchProgress(batchIdx, totalBatches, currentLoss, samplesPerSec, etaSeconds, fractions);
}

//===================================================================================================================//

template <typename CoreT, typename CoreConfigT>
void NN_CLI::Runner<CoreT, CoreConfigT>::notifyEpochCompleted(int epochIdx, int totalEpochs, float epochLoss,
                                                              bool hasValLoss, float valLoss,
                                                              const std::string& summary)
{
  for (auto* observer : this->observers)
    observer->onEpochCompleted(epochIdx, totalEpochs, epochLoss, hasValLoss, valLoss, summary);
}

//===================================================================================================================//

template <typename CoreT, typename CoreConfigT>
void NN_CLI::Runner<CoreT, CoreConfigT>::notifyTrainingFinished(bool success, const std::string& finalSummary)
{
  for (auto* observer : this->observers)
    observer->onTrainingFinished(success, finalSummary);
}

//===================================================================================================================//

template <typename CoreT, typename CoreConfigT>
void NN_CLI::Runner<CoreT, CoreConfigT>::notifyModelInfoUpdated(const std::string& property, const std::string& value)
{
  for (auto* observer : this->observers)
    observer->onModelInfoUpdated(property, value);
}

//===================================================================================================================//

template <typename CoreT, typename CoreConfigT>
void NN_CLI::Runner<CoreT, CoreConfigT>::notifyLogMessage(const std::string& message, bool isError)
{
  for (auto* observer : this->observers)
    observer->onLogMessage(message, isError);
}

//===================================================================================================================//

template <typename CoreT, typename CoreConfigT>
void NN_CLI::Runner<CoreT, CoreConfigT>::notifyTimingUpdated(const std::string& metric, float value)
{
  for (auto* observer : this->observers)
    observer->onTimingUpdated(metric, value);
}

//===================================================================================================================//
//  Training progress handling
//===================================================================================================================//

template <typename CoreT, typename CoreConfigT>
void NN_CLI::Runner<CoreT, CoreConfigT>::handleTrainingProgress(const Common::TrainingProgressEvent<float>& progress,
                                                                ulong batchSize)
{
  QMutexLocker<QMutex> lock(&this->callbackMutex);

  bool isEpochComplete = progress.epochLoss > 0;

  if (isEpochComplete)
    this->lastEpochLoss = progress.epochLoss;

  // Reset the per-epoch sub-line statistics when a new epoch begins.
  if (!isEpochComplete && this->statsEpoch != static_cast<int>(progress.currentEpoch)) {
    this->statsEpoch = static_cast<int>(progress.currentEpoch);
    this->epochStartTime = std::chrono::steady_clock::now();
    this->runningLossSum = 0.0;
    this->runningLossCount = 0;
    this->rateWindow.clear();
  }

  // Observer notification — batch progress (per-GPU fractions).
  std::vector<float> fractions;
  int totalGPUs = progress.totalGPUs;

  if (totalGPUs > 1) {
    // Multi-GPU: reset per-epoch tracking at epoch boundaries.
    if (this->trackedEpoch != static_cast<int>(progress.currentEpoch) || this->trackedTotalGPUs != totalGPUs) {
      this->trackedEpoch = static_cast<int>(progress.currentEpoch);
      this->trackedTotalGPUs = totalGPUs;
      this->gpuFractions.assign(totalGPUs, 0.0f);
    }

    // Update this GPU's fraction.  The core reports currentSample as the
    // GPU's own cumulative sample count within the epoch (not a global
    // counter), and each GPU processes ~totalSamples/totalGPUs of them.
    if (progress.gpuIndex >= 0 && progress.gpuIndex < totalGPUs) {
      ulong samplesPerGPU = progress.totalSamples / totalGPUs;
      float gpuFraction = (samplesPerGPU > 0)
                            ? static_cast<float>(progress.currentSample) / static_cast<float>(samplesPerGPU)
                            : 0.0f;
      this->gpuFractions[progress.gpuIndex] = std::min(1.0f, std::max(0.0f, gpuFraction));
    }

    fractions = this->gpuFractions;
  } else {
    // Single GPU or CPU: single fraction.
    float fraction = (progress.totalSamples > 0)
                       ? static_cast<float>(progress.currentSample) / static_cast<float>(progress.totalSamples)
                       : 0.0f;
    fractions = {fraction};
  }

  // Accumulate the running average loss from per-sample losses.
  if (!isEpochComplete) {
    this->runningLossSum += static_cast<double>(progress.sampleLoss);
    this->runningLossCount++;
  }

  // Throttle mid-epoch observer notifications to ~progressReports per epoch;
  // epoch-complete events always pass through.
  if (!isEpochComplete) {
    if (this->ioConfig.progressReports == 0)
      return;

    ulong interval = std::max(static_cast<ulong>(1), progress.totalSamples / this->ioConfig.progressReports);

    if (progress.currentSample % interval != 0 && progress.currentSample != progress.totalSamples)
      return;
  }

  // Overall fraction of the epoch done (average across devices).
  double fractionDone = 0.0;

  for (float f : fractions)
    fractionDone += f;

  if (!fractions.empty())
    fractionDone /= static_cast<double>(fractions.size());

  double samplesDone = fractionDone * static_cast<double>(progress.totalSamples);
  auto now = std::chrono::steady_clock::now();

  // Record a rate sample for the sliding-window ingestion-rate estimate.
  if (!isEpochComplete) {
    this->rateWindow.push_back({samplesDone, now});

    ulong windowSize = std::max(static_cast<ulong>(2), batchSize / 2);

    while (this->rateWindow.size() > windowSize)
      this->rateWindow.pop_front();
  }

  // Ingestion rate from the sliding window (full-epoch fallback) and ETA.
  double rate = 0.0;

  if (this->rateWindow.size() >= 2) {
    const RateSample& oldest = this->rateWindow.front();
    const RateSample& newest = this->rateWindow.back();
    double sampleDelta = newest.samplesDone - oldest.samplesDone;
    double timeDelta = std::chrono::duration<double>(newest.timestamp - oldest.timestamp).count();
    rate = (timeDelta > 0.0) ? sampleDelta / timeDelta : 0.0;
  } else {
    double elapsed = std::chrono::duration<double>(now - this->epochStartTime).count();
    rate = (elapsed > 0.0) ? samplesDone / elapsed : 0.0;
  }

  double eta = (rate > 0.0) ? (static_cast<double>(progress.totalSamples) - samplesDone) / rate : 0.0;

  float currentLoss = isEpochComplete
                        ? progress.epochLoss
                        : (this->runningLossCount > 0
                             ? static_cast<float>(this->runningLossSum / static_cast<double>(this->runningLossCount))
                             : 0.0f);

  int batchIdx = static_cast<int>(progress.currentSample / batchSize);
  int totalBatches = static_cast<int>((progress.totalSamples + batchSize - 1) / batchSize);
  this->notifyBatchProgress(batchIdx, totalBatches, currentLoss, static_cast<float>(rate),
                            static_cast<float>(eta), fractions);
}

//===================================================================================================================//
//  Accessors
//===================================================================================================================//

template <typename CoreT, typename CoreConfigT>
int NN_CLI::Runner<CoreT, CoreConfigT>::getTotalEpochs() const
{
  return static_cast<int>(this->coreConfig.trainingConfig.numEpochs);
}

//===================================================================================================================//

template <typename CoreT, typename CoreConfigT>
const CoreConfigT& NN_CLI::Runner<CoreT, CoreConfigT>::getCoreConfig() const
{
  return this->coreConfig;
}

//===================================================================================================================//
//  Model info string builders
//===================================================================================================================//

template <typename CoreT, typename CoreConfigT>
std::string NN_CLI::Runner<CoreT, CoreConfigT>::getAugmentationString() const
{
  if (this->augConfig.augmentationFactor > 0 || this->augConfig.balanceAugmentation ||
      this->augConfig.fullAugmentation) {
    std::vector<std::string> parts;

    if (this->augConfig.fullAugmentation)
      parts.push_back("all-images");

    if (this->augConfig.transforms.horizontalFlip)
      parts.push_back("flip");

    if (this->augConfig.transforms.rotation > 0)
      parts.push_back("rot " + std::to_string(static_cast<int>(this->augConfig.transforms.rotation)) + "\xC2\xB0");

    if (this->augConfig.transforms.translation > 0)
      parts.push_back("trans " +
                      std::to_string(static_cast<int>(this->augConfig.transforms.translation * 100)) + "%");

    if (this->augConfig.transforms.brightness > 0)
      parts.push_back("bright " +
                      std::to_string(static_cast<int>(this->augConfig.transforms.brightness * 100)) + "%");

    if (this->augConfig.transforms.contrast > 0)
      parts.push_back("contrast " +
                      std::to_string(static_cast<int>(this->augConfig.transforms.contrast * 100)) + "%");

    if (this->augConfig.transforms.gaussianNoise > 0) {
      std::ostringstream oss;
      oss << "noise " << this->augConfig.transforms.gaussianNoise;
      parts.push_back(oss.str());
    }

    if (this->augConfig.transforms.randomErasing > 0)
      parts.push_back("erase " +
                      std::to_string(static_cast<int>(this->augConfig.transforms.randomErasing * 100)) + "%");

    if (this->augConfig.transforms.hueShift > 0)
      parts.push_back("hue " + std::to_string(static_cast<int>(this->augConfig.transforms.hueShift * 100)) + "%");

    if (this->augConfig.transforms.scaling > 0)
      parts.push_back("scale " + std::to_string(static_cast<int>(this->augConfig.transforms.scaling * 100)) + "%");

    if (this->augConfig.transforms.elasticDeformation.alpha > 0)
      parts.push_back("elastic");

    if (parts.empty()) {
      return "None";
    } else {
      std::string augStr;
      for (ulong i = 0; i < parts.size(); i++) {
        if (i > 0)
          augStr += ", ";
        augStr += parts[i];
      }
      return augStr;
    }
  } else {
    return "None";
  }
}

//===================================================================================================================//

template <typename CoreT, typename CoreConfigT>
std::string NN_CLI::Runner<CoreT, CoreConfigT>::getValidationString() const
{
  if (this->_numValidationSamples > 0) {
    float validationRatio = static_cast<float>(this->_numValidationSamples) /
                            static_cast<float>(this->_numOriginalTrainSamples + this->_numValidationSamples);
    bool validationAuto = this->augConfig.validationConfig.autoSize;
    std::ostringstream oss;
    oss << SummaryTable::formatWithCommas(this->_numValidationSamples) << " (" << std::fixed << std::setprecision(2)
        << (validationRatio * 100) << "%" << (validationAuto ? ", auto" : "") << ")";
    return oss.str();
  } else {
    return "Disabled";
  }
}

//===================================================================================================================//

template <typename CoreT, typename CoreConfigT>
std::string NN_CLI::Runner<CoreT, CoreConfigT>::getClassWeightsString() const
{
  const auto& weights = this->coreConfig.costFunctionConfig.weights;

  if (weights.empty()) {
    return "Uniform";
  } else {
    std::ostringstream oss;

    if (this->augConfig.autoClassWeights)
      oss << "Auto ";

    oss << "[";

    for (ulong i = 0; i < weights.size(); i++) {
      if (i > 0)
        oss << ", ";
      oss << std::fixed << std::setprecision(2) << weights[i];
    }

    oss << "]";
    return oss.str();
  }
}

//===================================================================================================================//
//  Model info row builder
//===================================================================================================================//

template <typename CoreT, typename CoreConfigT>
std::vector<NN_CLI::SummaryRow> NN_CLI::Runner<CoreT, CoreConfigT>::buildModelInfoRows() const
{
  std::vector<SummaryRow> rows;

  //-- Device --//
  std::string deviceStr =
    SummaryTable::deviceString(this->coreConfig.deviceType, this->coreConfig.numGPUs, this->coreConfig.numThreads);
  rows.push_back({"Device", deviceStr});

  //-- Input shape (only when non-empty, i.e. CNN) --//
  std::string inputShapeStr = this->getInputShapeString();
  if (!inputShapeStr.empty())
    rows.push_back({"Input shape", inputShapeStr});

  //-- Network type --//
  rows.push_back({"Network type", this->getNetworkType()});

  //-- Separator --//
  rows.push_back({"", ""});

  //-- Layer counts (conv / residual are CNN-only; omit the misleading "0"
  //   rows for plain ANN models) --//
  ulong numConvLayers = this->getNumConvLayers();
  if (numConvLayers > 0)
    rows.push_back({"Conv layers", std::to_string(numConvLayers)});

  rows.push_back({"Dense layers", std::to_string(this->getNumDenseLayers())});

  ulong numResidualBlocks = this->getNumResidualBlocks();
  if (numResidualBlocks > 0)
    rows.push_back({"Residual blocks", std::to_string(numResidualBlocks)});

  //-- Total parameters --//
  rows.push_back({"Total parameters", SummaryTable::formatWithCommas(this->getTotalParameters())});

  //-- Separator --//
  rows.push_back({"", ""});

  //-- Sample counts --//
  ulong totalSamples = this->_numOriginalTrainSamples + this->_numValidationSamples;
  rows.push_back({"Total samples", SummaryTable::formatWithCommas(totalSamples)});

  if (this->_numTrainSamples != this->_numOriginalTrainSamples) {
    ulong numAugmented = this->_numTrainSamples - this->_numOriginalTrainSamples;
    rows.push_back({"Training samples", SummaryTable::formatWithCommas(this->_numOriginalTrainSamples) + " + " +
                                          SummaryTable::formatWithCommas(numAugmented) +
                                          " augmented = " + SummaryTable::formatWithCommas(this->_numTrainSamples)});
  } else {
    rows.push_back({"Training samples", SummaryTable::formatWithCommas(this->_numTrainSamples)});
  }

  rows.push_back({"Validation samples", this->getValidationString()});

  //-- Augmentation (only when active) --//
  std::string augStr = this->getAugmentationString();
  if (augStr != "None")
    rows.push_back({"Augmentation", augStr});

  //-- Class weights --//
  rows.push_back({"Class weights", this->getClassWeightsString()});

  //-- Separator --//
  rows.push_back({"", ""});

  //-- Training config --//
  const auto& tc = this->coreConfig.trainingConfig;
  rows.push_back({"Epochs", std::to_string(tc.numEpochs)});
  rows.push_back({"Batch size", std::to_string(tc.batchSize)});

  std::ostringstream lrOss;
  lrOss << tc.learningRate;
  rows.push_back({"Learning rate", lrOss.str()});

  std::string optStr = Common::Optimizer<float>::typeToName(tc.optimizer.type);
  optStr[0] = toupper(optStr[0]);
  rows.push_back({"Optimizer", optStr});

  if (tc.dropoutRate > 0)
    rows.push_back({"Dropout", std::to_string(static_cast<int>(tc.dropoutRate * 100)) + "%"});

  //-- Cost function --//
  std::string costStr;
  switch (this->coreConfig.costFunctionConfig.type) {
  case Common::CostFunctionType::CROSS_ENTROPY:
    costStr = "Cross-entropy";
    break;
  case Common::CostFunctionType::SQUARED_DIFFERENCE:
    costStr = "Squared difference";
    break;
  case Common::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE:
    costStr = "Weighted squared difference";
    break;
  }
  rows.push_back({"Cost function", costStr});

  rows.push_back({"Shuffle", tc.shuffleSamples ? "Yes" : "No"});

  return rows;
}

//===================================================================================================================//
//  Shared methods
//===================================================================================================================//

template <typename CoreT, typename CoreConfigT>
NN_CLI::ValidationMetadata NN_CLI::Runner<CoreT, CoreConfigT>::buildValidationMetadata() const
{
  return {this->validationState.enabled, this->validationState.numValSamples, this->validationState.lastValLoss,
          this->validationState.bestValLoss, this->validationState.bestValEpoch};
}

//===================================================================================================================//

template <typename CoreT, typename CoreConfigT>
int NN_CLI::Runner<CoreT, CoreConfigT>::finishTraining(const QString& inputFilePath)
{
  // Defensive: unreachable in normal flow (train() clears loadedEpochHistory after
  // prepending), kept as safety net against future refactoring.
  if (!this->coreConfig.loadedEpochHistory.empty()) {
    this->core->prependEpochHistory(this->coreConfig.loadedEpochHistory);
    this->coreConfig.loadedEpochHistory.clear();
  }

  // Every epoch — including the last — is finalized by the epoch-completed
  // callback (validation, best-model save, history record), so there is no
  // end-of-run fix-up to do here; just persist the final model.

  const auto& trainingMetadata = this->core->getTrainingMetadata();
  ulong numEpochs = trainingMetadata.epochHistory.empty() ? this->coreConfig.trainingConfig.numEpochs
                                                          : trainingMetadata.epochHistory.back().epoch + 1;

  std::string summary = "Epochs: " + std::to_string(numEpochs) +
                        " | Samples: " + std::to_string(trainingMetadata.numSamples) +
                        " | Final loss: " + std::to_string(trainingMetadata.finalLoss);

  this->notifyTrainingFinished(true, summary);

  return finishTrainingCommon(this->logLevel, this->parser, inputFilePath, *this->core,
                              [this](const std::string& path) { this->doSaveModel(path); });
}

//===================================================================================================================//
//  Explicit template instantiations
//===================================================================================================================//

template class NN_CLI::Runner<ANN::Core<float>, ANN::CoreConfig<float>>;
template class NN_CLI::Runner<CNN::Core<float>, CNN::CoreConfig<float>>;
