#include "NN-CLI_Runner.hpp"

#include "ANN/ANN_Core.hpp"
#include "CNN/CNN_Core.hpp"

#include <algorithm>
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
void NN_CLI::Runner<CoreT, CoreConfigT>::notifyBatchProgress(int batchIdx, int totalBatches, float currentLoss,
                                                             float fraction)
{
  for (auto* observer : this->observers)
    observer->onBatchProgress(batchIdx, totalBatches, currentLoss, fraction);
}

//===================================================================================================================//

template <typename CoreT, typename CoreConfigT>
void NN_CLI::Runner<CoreT, CoreConfigT>::notifyEpochCompleted(int epochIdx, int totalEpochs, float epochLoss,
                                                              float accuracy, const std::string& summary)
{
  for (auto* observer : this->observers)
    observer->onEpochCompleted(epochIdx, totalEpochs, epochLoss, accuracy, summary);
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
