#include "NN-CLI_RunnerBase.hpp"

#include "NN-CLI_SummaryTable.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>

//===================================================================================================================//

NN_CLI::RunnerBase::RunnerBase(const QCommandLineParser& parser, NN_CLI::LogLevel logLevel, NN_CLI::IOConfig& ioConfig,
                               NN_CLI::AugmentationConfig& augConfig, const QString& configPath)
  : parser(parser),
    logLevel(logLevel),
    configPath(configPath),
    ioConfig(ioConfig),
    augConfig(augConfig)
{
}

//===================================================================================================================//

float NN_CLI::RunnerBase::getCurrentLearningRate() const
{
  return this->learningRateSchedulerState.currentLearningRate;
}

//===================================================================================================================//

const NN_CLI::IOConfig& NN_CLI::RunnerBase::getIOConfig() const
{
  return this->ioConfig;
}

//===================================================================================================================//

std::string NN_CLI::RunnerBase::getAugmentationString() const
{
  if (this->augConfig.augmentationFactor > 0 || this->augConfig.balanceAugmentation ||
      this->augConfig.fullAugmentation) {
    std::vector<std::string> parts;

    if (this->augConfig.fullAugmentation)
      parts.push_back("all-images");

    if (this->augConfig.transforms.horizontalFlip)
      parts.push_back("flip");

    if (this->augConfig.transforms.rotation > 0)
      parts.push_back("rot: " + std::to_string(static_cast<int>(this->augConfig.transforms.rotation)) + "\xC2\xB0");

    if (this->augConfig.transforms.translation > 0)
      parts.push_back("trans: " + std::to_string(static_cast<int>(this->augConfig.transforms.translation * 100)) + "%");

    if (this->augConfig.transforms.brightness > 0)
      parts.push_back("bright: " + std::to_string(static_cast<int>(this->augConfig.transforms.brightness * 100)) + "%");

    if (this->augConfig.transforms.contrast > 0)
      parts.push_back("contrast: " + std::to_string(static_cast<int>(this->augConfig.transforms.contrast * 100)) + "%");

    if (this->augConfig.transforms.gaussianNoise > 0) {
      std::ostringstream oss;
      oss << "noise: " << this->augConfig.transforms.gaussianNoise;
      parts.push_back(oss.str());
    }

    if (this->augConfig.transforms.randomErasing > 0)
      parts.push_back("erase: " + std::to_string(static_cast<int>(this->augConfig.transforms.randomErasing * 100)) +
                      "%");

    if (this->augConfig.transforms.hueShift > 0)
      parts.push_back("hue: " + std::to_string(static_cast<int>(this->augConfig.transforms.hueShift * 100)) + "%");

    if (this->augConfig.transforms.scaling > 0)
      parts.push_back("scale: " + std::to_string(static_cast<int>(this->augConfig.transforms.scaling * 100)) + "%");

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

std::string NN_CLI::RunnerBase::getValidationString() const
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

void NN_CLI::RunnerBase::handleTrainProgress(const Common::TrainProgressEvent<float>& progress, ulong batchSize)
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

  // Batch progress (per-GPU fractions).
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
      float gpuFraction =
        (samplesPerGPU > 0) ? static_cast<float>(progress.currentSample) / static_cast<float>(samplesPerGPU) : 0.0f;
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

  // Throttle mid-epoch notifications to ~progressReports per epoch;
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
  emit this->batchProgress(batchIdx, totalBatches, currentLoss, static_cast<float>(rate), static_cast<float>(eta),
                           fractions);
}

//===================================================================================================================//

NN_CLI::ValidationMetadata NN_CLI::RunnerBase::buildValidationMetadata() const
{
  return {this->validationState.enabled, this->validationState.numValSamples, this->validationState.lastValLoss,
          this->validationState.bestValidationLoss, this->validationState.bestValEpoch};
}
