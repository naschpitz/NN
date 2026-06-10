#ifndef NN_CLI_RUNNER_HPP
#define NN_CLI_RUNNER_HPP

#include "NN-CLI_AugmentationConfig.hpp"
#include "NN-CLI_IOConfig.hpp"
#include "NN-CLI_LogLevel.hpp"
#include "NN-CLI_ModelSerializer.hpp"
#include "NN-CLI_ProgressBar.hpp"
#include "NN-CLI_RunnerUtils.hpp"
#include "NN-CLI_TerminalUI.hpp"
#include "NN-CLI_TrainingTui.hpp"
#include "NN-CLI_Utils.hpp"

#include <QCommandLineParser>

#include <memory>
#include <mutex>
#include <string>

//===================================================================================================================//

namespace NN_CLI
{

  /**
   * Base class for ANNRunner and CNNRunner.  Holds all shared state and
   * provides buildValidationMetadata() and finishTraining().  Derived
   * classes supply the core-specific doSaveModel() implementation.
   */
  template <typename CoreT, typename CoreConfigT>
  class Runner
  {
    public:
      //-- Constructors --//
      Runner(const QCommandLineParser& parser, LogLevel logLevel, IOConfig& ioConfig, AugmentationConfig& augConfig,
                 std::unique_ptr<CoreT>& core, CoreConfigT& coreConfig);

      virtual ~Runner() = default;

    protected:
      //-- Methods --//
      ValidationMetadata buildValidationMetadata() const;
      int finishTraining(const QString& inputFilePath);

      //-- Pure virtual --//
      virtual void doSaveModel(const std::string& outputPath) = 0;

      //-- Shared state --//
      const QCommandLineParser& parser;
      LogLevel logLevel;
      IOConfig& ioConfig;
      AugmentationConfig& augConfig;
      std::unique_ptr<CoreT>& core;
      CoreConfigT& coreConfig;

      //-- Validation state --//
      ValidationState validationState;

      //-- Callback state --//
      // Latest epoch-average training loss, written by the per-batch progress
      // callback and read by the epoch-completed callback.
      float lastEpochLoss = 0.0f;
      // Serializes the per-batch progress callback (fired concurrently from
      // worker threads) against the epoch-completed callback.
      std::mutex callbackMutex;
      std::unique_ptr<ProgressBar> progressBar;

      //-- ncurses terminal UI (only active during training) --//
      std::shared_ptr<TerminalUI> tui;

      //-- Loading-bar wiring shared with the CNN runner --//
      TrainingTui trainingTui;
  };

} // namespace NN_CLI

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

#endif // NN_CLI_RUNNER_HPP
