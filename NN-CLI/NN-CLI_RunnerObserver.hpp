#ifndef NN_CLI_RUNNEROBSERVER_HPP
#define NN_CLI_RUNNEROBSERVER_HPP

#include "NN-CLI_Types.hpp"

#include <string>
#include <vector>

//===================================================================================================================//

namespace NN_CLI
{

  /**
   * Observer interface for Runner training events.  Concrete observers
   * (e.g. a Controller bridging to the TerminalUI view) implement these
   * methods to receive notifications without the Runner depending on the
   * View layer directly.
   *
   * All methods have empty default implementations so observers can opt
   * into only the events they care about.
   */
  class IRunnerObserver
  {
    public:
      //-- Constructors --//
      virtual ~IRunnerObserver() = default;

      //-- Training lifecycle events --//

      /**
       * Called after each mini-batch is processed during training.
       * @param batchIdx       0-based index of the current mini-batch within the epoch.
       * @param totalBatches   Total number of mini-batches in the epoch.
       * @param currentLoss    Running average loss for the current epoch.
       * @param fractions      Per-device progress fractions [0..1].  One element
       *                       for single-GPU/CPU; one per GPU for multi-GPU training.
       */
      virtual void onBatchProgress(int batchIdx, int totalBatches, float currentLoss,
                                   const std::vector<float>& fractions) {}

      /**
       * Called while a batch of samples is being loaded/augmented from the
       * data source (drives the "Samples" data-loading progress bar).
       * @param current        Number of samples loaded so far in this batch.
       * @param total          Total samples in this batch.
       * @param batchIndex     0-based index of the batch currently loading.
       * @param totalBatches   Total number of batches in the epoch.
       * @param isValidation   True when loading validation samples.
       */
      virtual void onSampleLoadProgress(ulong current, ulong total, ulong batchIndex, ulong totalBatches,
                                        bool isValidation)
      {
        (void)current;
        (void)total;
        (void)batchIndex;
        (void)totalBatches;
        (void)isValidation;
      }

      /**
       * Called while the validation pass runs at an epoch boundary (drives the
       * transitional "Validating" progress bar).
       * @param current        Number of validation samples evaluated so far.
       * @param total          Total validation samples.
       */
      virtual void onValidationProgress(ulong current, ulong total)
      {
        (void)current;
        (void)total;
      }

      /**
       * Called once per epoch after all mini-batches are processed.
       * @param epochIdx       0-based index of the completed epoch.
       * @param totalEpochs    Total number of epochs in the training run.
       * @param epochLoss      Average loss over the completed epoch.
       * @param accuracy       Validation accuracy if available, else -1.0f.
       * @param summary        Human-readable epoch summary string.
       */
      virtual void onEpochCompleted(int epochIdx, int totalEpochs, float epochLoss, float accuracy,
                                    const std::string& summary)
      {
        (void)epochIdx;
        (void)totalEpochs;
        (void)epochLoss;
        (void)accuracy;
        (void)summary;
      }

      /**
       * Called when the entire training run finishes (or fails).
       * @param success        True if training completed normally.
       * @param finalSummary   Human-readable summary of the full run.
       */
      virtual void onTrainingFinished(bool success, const std::string& finalSummary)
      {
        (void)success;
        (void)finalSummary;
      }

      //-- Model / informational events --//

      /**
       * Called when a model property is updated (e.g. config display).
       * @param property       Name of the property.
       * @param value          String representation of the property value.
       */
      virtual void onModelInfoUpdated(const std::string& property, const std::string& value)
      {
        (void)property;
        (void)value;
      }

      /**
       * Called for general-purpose log messages (replaces direct
       * std::cout / std::cerr calls).
       * @param message        The log message.
       * @param isError        True if this is an error message.
       */
      virtual void onLogMessage(const std::string& message, bool isError)
      {
        (void)message;
        (void)isError;
      }

      /**
       * Called when a timing metric is updated (e.g. per-phase profiling).
       * @param metric         Name of the timing metric.
       * @param value          Timing value (typically in milliseconds or seconds).
       */
      virtual void onTimingUpdated(const std::string& metric, float value)
      {
        (void)metric;
        (void)value;
      }
  };

} // namespace NN_CLI

//===================================================================================================================//

#endif // NN_CLI_RUNNEROBSERVER_HPP
