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

#include "Common/Common_TrainingProgressEvent.hpp"

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
      virtual ulong getNumConvLayers() const { return 0; }

      // Return the number of dense (fully-connected) layers.
      virtual ulong getNumDenseLayers() const = 0;

      // Return the number of residual blocks (default 0 for ANN).
      virtual ulong getNumResidualBlocks() const { return 0; }

      //-- Sample-count accessors --//

      // Return the number of original training samples (before augmentation).
      ulong getNumOriginalTrainSamples() const { return _numOriginalTrainSamples; }

      // Return the total number of training samples (after augmentation).
      ulong getNumTrainSamples() const { return _numTrainSamples; }

      // Return the number of validation samples (0 if validation is disabled).
      ulong getNumValidationSamples() const { return _numValidationSamples; }

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

      //-- Model info row builder --//

      // Build the complete set of SummaryRows describing the model
      // configuration and training setup, mirroring the order and separators
      // used by TrainingSummary::collectCNNRows / collectRows.
      std::vector<SummaryRow> buildModelInfoRows() const;

    protected:
      //-- Methods --//
      ValidationMetadata buildValidationMetadata() const;
      int finishTraining(const QString& inputFilePath);

      // Shared per-batch training-progress handler, installed as the core's
      // training callback by both ANNRunner and CNNRunner: tracks per-GPU
      // fractions (reset at epoch boundaries) and notifies observers of
      // batch progress.  Thread-safe (locks callbackMutex).
      void handleTrainingProgress(const Common::TrainingProgressEvent<float>& progress, ulong batchSize);

      //-- Pure virtual --//
      virtual void doSaveModel(const std::string& outputPath) = 0;

      //-- Observer notifications --//
      void notifySampleLoadProgress(ulong current, ulong total, ulong batchIndex, ulong totalBatches,
                                    bool isValidation);
      void notifyValidationProgress(ulong current, ulong total);
      void notifyBatchProgress(int batchIdx, int totalBatches, float currentLoss, float samplesPerSec,
                               float etaSeconds, const std::vector<float>& fractions);
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

      //-- Sample counts (set during train()) --//
      ulong _numOriginalTrainSamples = 0;
      ulong _numTrainSamples = 0;
      ulong _numValidationSamples = 0;

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
