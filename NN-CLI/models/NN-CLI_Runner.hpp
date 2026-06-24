#ifndef NN_CLI_RUNNER_HPP
#define NN_CLI_RUNNER_HPP

#include "NN-CLI_AugmentationConfig.hpp"
#include "NN-CLI_IOConfig.hpp"
#include "NN-CLI_LogLevel.hpp"
#include "NN-CLI_ModelSerializer.hpp"
#include "NN-CLI_RunnerObserver.hpp"
#include "NN-CLI_RunnerUtils.hpp"
#include "NN-CLI_SummaryTable.hpp"
#include "NN-CLI_Utils.hpp"

#include "Common/Common_LearningRateScheduler.hpp"
#include "Common/Common_TrainProgressEvent.hpp"

#include <QCommandLineParser>
#include <QMutex>

#include <chrono>
#include <deque>
#include <memory>
#include <string>
#include <vector>

//===================================================================================================================//

namespace NN_CLI
{

  /**
   * Base class for ANNRunner and CNNRunner.  Holds all shared state and
    * provides buildValidationMetadata() and finishTrain().  Derived
   * classes supply the core-specific doSaveModel() implementation.
   */
  template <typename CoreT, typename CoreConfigT>
  class Runner
  {
    public:
      //-- Constructors --//
      Runner(const QCommandLineParser& parser, LogLevel logLevel, IOConfig& ioConfig, AugmentationConfig& augConfig,
             std::unique_ptr<CoreT>& core, CoreConfigT& coreConfig, const QString& configPath);

      virtual ~Runner() = default;

      //-- Observer management --//
      void addObserver(IRunnerObserver* observer);
      void removeObserver(IRunnerObserver* observer);

      //-- Accessors --//

      // Return the total number of epochs configured for this training run.
      int getTotalEpochs() const;

      // Return the current learning rate (reflects learning-rate-scheduler adjustments
      // at epoch boundaries; equals trainConfig.learningRate when no scheduler).
      float getCurrentLearningRate() const;

      // Return a const reference to the core configuration.
      const CoreConfigT& getCoreConfig() const;

      // Return a const reference to the I/O configuration.
      const IOConfig& getIOConfig() const;

      // Request the core to stop (abort) the current training run.
      void requestAbort()
      {
        if (this->core)
          this->core->requestStop();
      }

      // Return formatted timing/profiling lines for display in the TUI timing panel.
      // maxWidth > 0 constrains output to the given column width; maxWidth == 0
      // auto-detects the terminal width.
      virtual std::vector<std::string> getTimingLines(int maxWidth = 0) const = 0;

      // Return the number of output classes (neurons in the final layer).
      virtual ulong getNumOutputClasses() const = 0;

      //-- Model info virtual accessors --//

      // Return the total number of trainable parameters in the network.
      virtual ulong getTotalParameters() const = 0;

      // Return a human-readable network type string (e.g. "ANN", "CNN").
      virtual std::string getNetworkType() const = 0;

      // Return a human-readable input shape string (e.g. "1 x 28 x 28" for
      // CNN, empty string for ANN where input shape is implicit).
      virtual std::string getInputShapeString() const = 0;

      // Return the number of convolutional layers (default 0 for ANN).
      virtual ulong getNumConvLayers() const
      {
        return 0;
      }

      // Return the number of dense (fully-connected) layers.
      virtual ulong getNumDenseLayers() const = 0;

      // Return the number of residual blocks (default 0 for ANN).
      virtual ulong getNumResidualBlocks() const
      {
        return 0;
      }

      //-- Sample-count accessors --//

      // Return the number of original training samples (before augmentation).
      ulong getNumOriginalTrainSamples() const
      {
        return _numOriginalTrainSamples;
      }

      // Return the total number of training samples (after augmentation).
      ulong getNumTrainSamples() const
      {
        return _numTrainSamples;
      }

      // Return the number of validation samples (0 if validation is disabled).
      ulong getNumValidationSamples() const
      {
        return _numValidationSamples;
      }

      //-- Model info string builders --//

      // Return a human-readable string describing the current augmentation
      // configuration (e.g. "flip, rot 15°" or "None").
      std::string getAugmentationString() const;

      // Return a human-readable string describing the validation split
      // (e.g. "1,000 (10.00%, auto)" or "Disabled").
      std::string getValidationString() const;

      // Return a human-readable string describing the class weights
      // (e.g. "Uniform" or "Auto [0.50, 1.20, 0.80]").
      std::string getClassWeightsString() const;

      // Return a human-readable string describing the learning-rate scheduler
      // configuration (e.g. "type: step, gamma: 0.1, step size: 3" or "None").
      std::string getLearningRateSchedulerString() const;

      //-- Model info row builder --//

      // Build the complete set of SummaryRows describing the model
      // configuration and training setup, mirroring the order and separators
      // used by TrainSummary::collectCNNRows / collectRows.
      std::vector<SummaryRow> buildModelInfoRows() const;

      // Build architecture-focused SummaryRows for the predict TerminalUI
      // model info panel (Device, Network type, layer counts, parameters,
      // saved training config).  Omits sample-count and augmentation rows.
      std::vector<SummaryRow> buildPredictModelInfoRows() const;

    protected:
      //-- Methods --//
      ValidationMetadata buildValidationMetadata() const;
      int finishTrain(const QString& inputFilePath);

      // Shared per-batch training-progress handler, installed as the core's
      // training callback by both ANNRunner and CNNRunner: tracks per-GPU
      // fractions (reset at epoch boundaries) and notifies observers of
      // batch progress.  Thread-safe (locks callbackMutex).
      void handleTrainProgress(const Common::TrainProgressEvent<float>& progress, ulong batchSize);

      // Install a progress callback on the core that notifies observers of
      // batch progress during predict.  Mirrors the throttling/threshold
      // behavior of `Utils::setupModeProgressCallback` so it is a drop-in
      // replacement.  When `logLevel > QUIET` installs a callback that
      // calls `notifyBatchProgress` with a fraction derived from the
      // core's sample counter.
      void setupPredictProgressCallback(ulong total);

      // Advance the learning-rate scheduler one step at an epoch boundary and publish the
      // new learning rate to the core (no-op when scheduler.type == NONE).  `epoch` is the
      // 0-based index of the just-completed epoch (relative to this run); the
      // absolute index used by step/cosine is startingEpoch + epoch so curves
      // continue across resumes.  Called from both ANN/CNN epoch-completed
      // callbacks after validation loss is known.
      void applyLearningRateScheduler(ulong epoch, int totalEpochs, bool hasValLoss, float valLoss);

      //-- Pure virtual --//
      virtual void doSaveModel(const std::string& outputPath) = 0;

      //-- Observer notifications --//
      void notifySampleLoadProgress(ulong current, ulong total, ulong batchIndex, ulong totalBatches,
                                    bool isValidation);
      void notifyValidationProgress(ulong current, ulong total);
      void notifyBatchProgress(int batchIdx, int totalBatches, float currentLoss, float samplesPerSec, float etaSeconds,
                               const std::vector<float>& fractions);
      void notifyEpochCompleted(int epochIdx, int totalEpochs, float epochLoss, bool hasValLoss, float valLoss,
                                const std::string& summary);
      void notifyTrainFinished(bool success, const std::string& finalSummary);
      void notifyPredictFinished(const Common::PredictResults<float>& results, size_t numInputs, double durationSeconds,
                                 const std::string& durationFormatted, const std::string& outputPath);
      void notifyModelInfoUpdated(const std::string& property, const std::string& value);
      void notifyLogMessage(const std::string& message, bool isError);
      void notifyTimingUpdated(const std::string& metric, float value);

      //-- Shared state --//
      const QCommandLineParser& parser;
      LogLevel logLevel;
      QString configPath;
      IOConfig& ioConfig;
      AugmentationConfig& augConfig;
      std::unique_ptr<CoreT>& core;
      CoreConfigT& coreConfig;

      //-- Sample counts (set during train()) --//
      ulong _numOriginalTrainSamples = 0;
      ulong _numTrainSamples = 0;
      ulong _numValidationSamples = 0;

      //-- Validation state --//
      ValidationState validationState;

      //-- Learning-rate scheduler state (seeded in the ctor from trainConfig.learningRate;
      // overwritten on resume by the loaded checkpoint state). --//
      Common::LearningRateSchedulerState learningRateSchedulerState;

      //-- Callback state --//
      // Latest epoch-average training loss, written by the per-batch progress
      // callback and read by the epoch-completed callback.
      float lastEpochLoss = 0.0f;
      // Per-GPU fractions for multi-GPU training, reset at each epoch boundary.
      std::vector<float> gpuFractions;
      int trackedEpoch = -1;
      int trackedTotalGPUs = 0;
      // Progress sub-line statistics, reset at each epoch boundary: running
      // average sample loss and a sliding window of (samples done, timestamp)
      // pairs for the ingestion-rate / ETA estimate.
      double runningLossSum = 0.0;
      ulong runningLossCount = 0;
      int statsEpoch = -1;
      std::chrono::steady_clock::time_point epochStartTime;
      struct RateSample {
          double samplesDone;
          std::chrono::steady_clock::time_point timestamp;
      };

      std::deque<RateSample> rateWindow;
      // Serializes the per-batch progress callback (fired concurrently from
      // worker threads) against the epoch-completed callback.
      QMutex callbackMutex;

      //-- Observer list --//
      std::vector<IRunnerObserver*> observers;
  };

} // namespace NN_CLI

#endif // NN_CLI_RUNNER_HPP
