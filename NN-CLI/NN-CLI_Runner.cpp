#include "NN-CLI_Runner.hpp"

#include "ANN/ANN_Core.hpp"
#include "CNN/CNN_Core.hpp"

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
  return finishTrainingCommon(
    this->tui, this->logLevel, this->parser, inputFilePath, *this->core, [this](const std::string& path) {
      this->doSaveModel(path);
    });
}

//===================================================================================================================//
//  Explicit template instantiations
//===================================================================================================================//

template class NN_CLI::Runner<ANN::Core<float>, ANN::CoreConfig<float>>;
template class NN_CLI::Runner<CNN::Core<float>, CNN::CoreConfig<float>>;
