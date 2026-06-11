#ifndef NN_CLI_RUNNER_HPP
#define NN_CLI_RUNNER_HPP

#include "NN-CLI_AugmentationConfig.hpp"
#include "NN-CLI_IOConfig.hpp"
#include "NN-CLI_LogLevel.hpp"
#include "NN-CLI_ModelSerializer.hpp"
#include "NN-CLI_RunnerObserver.hpp"
#include "NN-CLI_RunnerUtils.hpp"
#include "NN-CLI_Utils.hpp"

#include <QCommandLineParser>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

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

      //-- Observer management --//
      void addObserver(IRunnerObserver* observer);
      void removeObserver(IRunnerObserver* observer);

      //-- Accessors --//

      // Return the total number of epochs configured for this training run.
      int getTotalEpochs() const;

      // Return a const reference to the core configuration.
      const CoreConfigT& getCoreConfig() const;

      // Return formatted timing/profiling lines for display in the TUI timing panel.
      // maxWidth > 0 constrains output to the given column width; maxWidth == 0
      // auto-detects the terminal width.
      virtual std::vector<std::string> getTimingLines(int maxWidth = 0) const = 0;

    protected:
      //-- Methods --//
      ValidationMetadata buildValidationMetadata() const;
      int finishTraining(const QString& inputFilePath);

      //-- Pure virtual --//
      virtual void doSaveModel(const std::string& outputPath) = 0;

      //-- Observer notifications --//
      void notifyBatchProgress(int batchIdx, int totalBatches, float currentLoss,
                               const std::vector<float>& fractions);
      void notifyEpochCompleted(int epochIdx, int totalEpochs, float epochLoss, float accuracy,
                                const std::string& summary);
      void notifyTrainingFinished(bool success, const std::string& finalSummary);
      void notifyModelInfoUpdated(const std::string& property, const std::string& value);
      void notifyLogMessage(const std::string& message, bool isError);
      void notifyTimingUpdated(const std::string& metric, float value);

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
      // Per-GPU fractions for multi-GPU training, reset at each epoch boundary.
      std::vector<float> gpuFractions;
      int trackedEpoch = -1;
      int trackedTotalGPUs = 0;
      // Serializes the per-batch progress callback (fired concurrently from
      // worker threads) against the epoch-completed callback.
      std::mutex callbackMutex;

      //-- Observer list --//
      std::vector<IRunnerObserver*> observers;
  };

} // namespace NN_CLI

#endif // NN_CLI_RUNNER_HPP
