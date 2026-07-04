#include "NN-CLI_Runner.hpp"

#include "ANN/ANN_Core.hpp"
#include "CNN/CNN_Core.hpp"

#include "NN-CLI_RunnerUtils.hpp"

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
                                           std::unique_ptr<CoreT>& core, CoreConfigT& coreConfig,
                                           const QString& configPath)
  : RunnerBase(parser, logLevel, ioConfig, augConfig, configPath),
    core(core),
    coreConfig(coreConfig)
{
  // Seed the learning-rate scheduler: the base/current learning rate is the configured learning rate.
  // On resume this is overwritten by the checkpoint's learningRateSchedulerState (Phase 6-state).
  this->learningRateSchedulerState.baseLearningRate = coreConfig.trainConfig.learningRate;
  this->learningRateSchedulerState.currentLearningRate = coreConfig.trainConfig.learningRate;

  // On resume, restore the persisted scheduler state so the schedule continues (plateau needs this).
  if (coreConfig.loadedTrainMetadata.learningRateSchedulerState.initialized)
    this->learningRateSchedulerState = coreConfig.loadedTrainMetadata.learningRateSchedulerState;

  // Wire the predict-progress forwarding once, for the Runner's lifetime, instead of per mode
  // call: a reused Runner would otherwise stack duplicate connections (QObject::connect on a
  // lambda is not idempotent and Qt::UniqueConnection does not dedupe lambdas). test() still
  // wires its own short-lived testCore.
  this->emitProgressFromCore(*this->core);
}

//===================================================================================================================//
//  Learning-rate scheduler
//===================================================================================================================//

template <typename CoreT, typename CoreConfigT>
void NN_CLI::Runner<CoreT, CoreConfigT>::applyLearningRateScheduler(ulong epoch, int totalEpochs, bool hasValLoss,
                                                                    float valLoss)
{
  const auto& trainConfig = this->core->getTrainConfig();
  const auto& cfg = trainConfig.learningRateScheduler;

  if (cfg.type == Common::LearningRateSchedulerType::NONE)
    return;

  // Mark state as live so it persists across checkpoint saves (enables plateau resume).
  this->learningRateSchedulerState.initialized = true;

  // Absolute epoch index keeps step/cosine curves continuous across resumed runs.
  const ulong absEpoch = trainConfig.startingEpoch + epoch;
  const float newLearningRate = Common::stepLearningRateScheduler(cfg, this->learningRateSchedulerState, absEpoch,
                                                                  static_cast<ulong>(totalEpochs), hasValLoss, valLoss);

  if (newLearningRate != this->learningRateSchedulerState.currentLearningRate) {
    this->core->setLearningRate(newLearningRate);
  }

  this->learningRateSchedulerState.currentLearningRate = newLearningRate;

  // Publish the state into the core's metadata so every save path serializes it.
  this->core->getTrainMetadata().learningRateSchedulerState = this->learningRateSchedulerState;
}

//===================================================================================================================//
//  Accessors
//===================================================================================================================//

template <typename CoreT, typename CoreConfigT>
int NN_CLI::Runner<CoreT, CoreConfigT>::getTotalEpochs() const
{
  return static_cast<int>(this->coreConfig.trainConfig.numEpochs);
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

template <typename CoreT, typename CoreConfigT>
std::string NN_CLI::Runner<CoreT, CoreConfigT>::getLearningRateSchedulerString() const
{
  const auto& sched = this->coreConfig.trainConfig.learningRateScheduler;

  if (sched.type == Common::LearningRateSchedulerType::NONE)
    return "None";

  auto formatFloat = [](float value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << value;
    std::string s = oss.str();
    auto dot = s.find('.');

    if (dot != std::string::npos) {
      s.erase(s.find_last_not_of('0') + 1, std::string::npos);

      if (s.back() == '.')
        s.pop_back();
    }

    return s;
  };

  std::vector<std::string> parts;
  parts.push_back("type: " + Common::LearningRateSchedulerConfig::typeToName(sched.type));

  // Show every field whose value differs from its default, in declaration order
  // (matches how augmentation lists only the active transforms).
  if (sched.gamma != 0.1f)
    parts.push_back("gamma: " + formatFloat(sched.gamma));

  if (sched.stepSize != 1)
    parts.push_back("step size: " + std::to_string(sched.stepSize));

  if (sched.minLearningRate != 0.0f)
    parts.push_back("minimum learning rate: " + formatFloat(sched.minLearningRate));

  if (sched.patience != 10)
    parts.push_back("patience: " + std::to_string(sched.patience));

  if (sched.minDelta != 1e-4f)
    parts.push_back("minimum delta: " + formatFloat(sched.minDelta));

  std::string result;

  for (ulong i = 0; i < parts.size(); i++) {
    if (i > 0)
      result += ", ";
    result += parts[i];
  }

  return result;
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
  const auto& tc = this->coreConfig.trainConfig;
  rows.push_back({"Epochs", std::to_string(tc.numEpochs)});
  rows.push_back({"Batch size", std::to_string(tc.batchSize)});

  std::ostringstream lrOss;
  lrOss << tc.learningRate;
  rows.push_back({"Learning rate", lrOss.str()});

  std::string optStr = Common::optimizerTypeToName(tc.optimizer.type);
  optStr[0] = toupper(optStr[0]);
  rows.push_back({"Optimizer", optStr});

  std::string schedulerStr = this->getLearningRateSchedulerString();

  if (schedulerStr != "None")
    rows.push_back({"Learning-rate scheduler", schedulerStr});

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

template <typename CoreT, typename CoreConfigT>
std::vector<NN_CLI::SummaryRow> NN_CLI::Runner<CoreT, CoreConfigT>::buildPredictModelInfoRows() const
{
  std::vector<SummaryRow> rows;

  //-- Device --//
  std::string deviceStr =
    SummaryTable::deviceString(this->coreConfig.deviceType, this->coreConfig.numGPUs, this->coreConfig.numThreads);
  rows.push_back({"Device", deviceStr});

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

  //-- Output classes --//
  rows.push_back({"Output classes", SummaryTable::formatWithCommas(this->getNumOutputClasses())});

  //-- Separator --//
  rows.push_back({"", ""});

  //-- Saved training config --//
  const auto& tc = this->coreConfig.trainConfig;
  rows.push_back({"Epochs", std::to_string(tc.numEpochs)});
  rows.push_back({"Batch size", std::to_string(tc.batchSize)});

  std::ostringstream lrOss;
  lrOss << tc.learningRate;
  rows.push_back({"Learning rate", lrOss.str()});

  std::string optStr = Common::optimizerTypeToName(tc.optimizer.type);
  optStr[0] = toupper(optStr[0]);
  rows.push_back({"Optimizer", optStr});

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

  return rows;
}

//===================================================================================================================//

template <typename CoreT, typename CoreConfigT>
void NN_CLI::Runner<CoreT, CoreConfigT>::emitProgressFromCore(CoreT& core)
{
  if (this->logLevel > LogLevel::QUIET) {
    auto& hub = core.getCoreSignals();

    QObject::connect(
      &hub, &std::remove_reference_t<decltype(hub)>::predictProgress, this,
      [this](ulong current, ulong total) {
        if (total == 0)
          return;

        // Throttle: only notify every total/progressReports samples.
        ulong reports = this->ioConfig.progressReports;
        ulong interval = (reports > 0) ? std::max(static_cast<ulong>(1), total / reports) : 0;

        if (interval == 0)
          return;

        if (current != total && (current % interval) != 0)
          return;

        // Compute batch-level progress.
        int batchIdx = static_cast<int>(current / this->coreConfig.trainConfig.batchSize);
        int totalBatches = static_cast<int>((total + this->coreConfig.trainConfig.batchSize - 1) /
                                            this->coreConfig.trainConfig.batchSize);

        float fraction = static_cast<float>(current) / static_cast<float>(total);

        emit this->batchProgress(batchIdx, totalBatches, 0.f, 0.f, 0.f, {fraction});
      },

      Qt::DirectConnection);
  }
}

//===================================================================================================================//
//  Shared methods
//===================================================================================================================//

template <typename CoreT, typename CoreConfigT>
int NN_CLI::Runner<CoreT, CoreConfigT>::finishTrain(const QString& inputFilePath)
{
  // Defensive: unreachable in normal flow (train() clears loadedTrainMetadata.epochHistory after
  // prepending), kept as safety net against future refactoring.
  if (!this->coreConfig.loadedTrainMetadata.epochHistory.empty()) {
    this->core->prependEpochHistory(this->coreConfig.loadedTrainMetadata.epochHistory);
    this->coreConfig.loadedTrainMetadata.epochHistory.clear();
  }

  // Every epoch — including the last — is finalized by the epoch-completed
  // callback (validation, best-model save, history record), so there is no
  // end-of-run fix-up to do here; just persist the final model.

  const auto& trainMetadata = this->core->getTrainMetadata();
  ulong numEpochs = trainMetadata.epochHistory.empty() ? this->coreConfig.trainConfig.numEpochs
                                                       : trainMetadata.epochHistory.back().epoch + 1;

  std::string summary = "Epochs: " + std::to_string(numEpochs) +
                        " | Samples: " + std::to_string(trainMetadata.numSamples) +
                        " | Final loss: " + std::to_string(trainMetadata.finalLoss);

  emit this->trainFinished(true, summary);

  return NN_CLI::RunnerUtils::finishTrainCommon(this->logLevel, this->parser, inputFilePath, *this->core,
                                                [this](const std::string& path) { this->doSaveModel(path); });
}

//===================================================================================================================//
//  Explicit template instantiations
//===================================================================================================================//

template class NN_CLI::Runner<ANN::Core<float>, ANN::CoreConfig<float>>;
template class NN_CLI::Runner<CNN::Core<float>, CNN::CoreConfig<float>>;
