#ifndef NN_CLI_RUNNER_HPP
#define NN_CLI_RUNNER_HPP

#include "NN-CLI_AugmentationConfig.hpp"
#include "NN-CLI_IOConfig.hpp"
#include "NN-CLI_LogLevel.hpp"
#include "NN-CLI_ModelSerializer.hpp"
#include "NN-CLI_RunnerUtils.hpp"
#include "NN-CLI_TerminalUI.hpp"
#include "NN-CLI_TrainingController.hpp"
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

      //-- ncurses terminal UI (only active during training) --//
      std::shared_ptr<TerminalUI> tui;

      //-- Loading-bar wiring shared with the CNN runner --//
      TrainingController trainingController;
  };

} // namespace NN_CLI


#endif // NN_CLI_RUNNER_HPP
