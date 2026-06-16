#include "NN-CLI_PredictController.hpp"

#include "NN-CLI_ANNRunner.hpp"
#include "NN-CLI_CNNRunner.hpp"

#include <QMutex>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
  //===================================================================================================================//

  // Right-pad a string with spaces to the given width.  If the string already
  // exceeds the width it is returned unchanged.
  std::string formatPadded(const std::string& s, int width)
  {
    if (static_cast<int>(s.size()) >= width)
      return s;

    return s + std::string(static_cast<std::string::size_type>(width - s.size()), ' ');
  }

} // namespace

namespace NN_CLI
{

  //===================================================================================================================//
  //-- Ctors / Dtors --//
  //===================================================================================================================//

  template <typename RunnerT>
  PredictController<RunnerT>::~PredictController()
  {
    if (this->runner)
      this->runner->removeObserver(this);
  }

  //===================================================================================================================//
  //-- Lifecycle --//
  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::init(std::unique_ptr<RunnerT> runner)
  {
    this->window_ = std::make_unique<TerminalUI_PredictWindow>();
    this->runner = std::move(runner);

    if (this->runner)
      this->runner->addObserver(this);

    // Initialize the ncurses TUI.  If init fails (e.g. no TTY attached),
    // the window gracefully degrades — the UI thread is never started, so
    // the prediction proceeds with console-only output from the Runner.
    if (this->window_ && this->window_->init()) {
      this->populateModelInfo();
      this->populateTrainMeta();
      this->populateProgress();
      this->window_->startUiThread();
    }
  }

  //===================================================================================================================//

  template <typename RunnerT>
  int PredictController<RunnerT>::startPredict()
  {
    if (!this->runner)
      return 1;

    int r = this->runner->predict();

    // Block on dismiss after predict completes when the TUI is active.
    if (this->window_ && this->window_->isInitialized())
      this->window_->waitForDismiss();

    return r;
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::shutdown()
  {
    if (this->runner)
      this->runner->removeObserver(this);

    this->window_.reset();
  }

  //===================================================================================================================//
  //-- Accessors --//
  //===================================================================================================================//

  template <typename RunnerT>
  RunnerT* PredictController<RunnerT>::getRunner() const
  {
    return this->runner.get();
  }

  //===================================================================================================================//

  template <typename RunnerT>
  TerminalUI_PredictWindow* PredictController<RunnerT>::getWindow() const
  {
    return this->window_.get();
  }

  //===================================================================================================================//
  //-- IRunnerObserver overrides --//
  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::onSampleLoadProgress(ulong current, ulong total, ulong batchIndex,
                                                        ulong totalBatches, bool isValidation)
  {
    (void)batchIndex;
    (void)totalBatches;
    (void)isValidation;

    if (!this->window_ || !this->window_->isInitialized())
      return;

    QMutexLocker<QRecursiveMutex> lock(&this->window_->getMutex());

    float fraction = (total > 0) ? static_cast<float>(current) / static_cast<float>(total) : 0.0f;

    this->window_->setLoadingProgress("Samples " + std::to_string(current) + "/" + std::to_string(total), fraction);
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::onValidationProgress(ulong current, ulong total)
  {
    if (!this->window_ || !this->window_->isInitialized())
      return;

    QMutexLocker<QRecursiveMutex> lock(&this->window_->getMutex());

    float fraction = (total > 0) ? static_cast<float>(current) / static_cast<float>(total) : 0.0f;
    this->window_->updateProgress("Validating", fraction);
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::onBatchProgress(int batchIdx, int totalBatches, float currentLoss,
                                                   float samplesPerSec, float etaSeconds,
                                                   const std::vector<float>& fractions)
  {
    (void)currentLoss;
    (void)samplesPerSec;
    (void)etaSeconds;

    if (!this->window_ || !this->window_->isInitialized()) {
      // Console fallback.
      float fraction = fractions.empty() ? 0.0f : fractions[0];
      std::cout << "\r  Progress: " << (batchIdx + 1) << "/" << totalBatches << " (" << std::fixed
                << std::setprecision(1) << (fraction * 100.0f) << "%)" << std::flush;
      return;
    }

    QMutexLocker<QRecursiveMutex> lock(&this->window_->getMutex());

    this->checkAbortRequested();

    float fraction = fractions.empty() ? 0.0f : fractions[0];
    this->window_->updateProgress("Predicting " + std::to_string(batchIdx + 1) + "/" + std::to_string(totalBatches),
                                  fraction);
    this->window_->updateProgressSubLine(std::to_string(batchIdx + 1) + "/" + std::to_string(totalBatches) + " (" +
                                         std::to_string(static_cast<int>(fraction * 100)) + "%)");
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::onEpochCompleted(int epochIdx, int totalEpochs, float epochLoss, bool hasValLoss,
                                                    float valLoss, const std::string& summary)
  {
    (void)epochIdx;
    (void)totalEpochs;
    (void)epochLoss;
    (void)hasValLoss;
    (void)valLoss;

    // When the TUI is not active, print to console for interface completeness.
    if (!this->window_ || !this->window_->isInitialized()) {
      std::cout << summary << "\n";
      return;
    }

    // When the TUI is active these events are informational only.
    QMutexLocker<QRecursiveMutex> lock(&this->window_->getMutex());
    this->window_->updateProgressSubLine(summary);
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::onTrainFinished(bool success, const std::string& finalSummary)
  {
    if (!this->window_ || !this->window_->isInitialized()) {
      std::string prefix = success ? "[Predict complete] " : "[Predict failed] ";
      std::cout << "\n" << prefix << finalSummary << "\n";
      return;
    }

    QMutexLocker<QRecursiveMutex> lock(&this->window_->getMutex());

    std::string prefix = success ? "[Predict complete] " : "[Predict failed] ";
    this->window_->updateProgressSubLine(prefix + finalSummary);
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::onPredictFinished(const Common::PredictResults<float>& results, size_t numInputs,
                                                     double durationSeconds, const std::string& durationFormatted,
                                                     const std::string& outputPath)
  {
    (void)durationSeconds;

    if (!this->window_ || !this->window_->isInitialized())
      return;

    QMutexLocker<QRecursiveMutex> lock(&this->window_->getMutex());

    const auto& ioConfig = this->runner->getIOConfig();

    // Image output: single summary line.
    if (ioConfig.outputType == DataType::IMAGE) {
      this->window_->addResultRow({"", std::to_string(numInputs) + " image outputs -> " + outputPath, ""});
    } else {
      // Vector output: one row per sample.
      for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];

        size_t predictedClass = 0;
        float confidence = 0.0f;

        for (size_t j = 0; j < r.output.size(); ++j) {
          if (r.output[j] > confidence) {
            confidence = r.output[j];
            predictedClass = j;
          }
        }

        std::string classStr = std::to_string(predictedClass);
        std::string pctStr = std::to_string(static_cast<int>(confidence * 100)) + "%";

        this->window_->addResultRow({std::to_string(i), classStr, pctStr});
      }
    }

    this->window_->refreshResultsContent();
    this->window_->updateProgress("Predicting", 1.0f);
    this->window_->updateProgressSubLine("Done — " + durationFormatted + ", output: " + outputPath);
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::onModelInfoUpdated(const std::string& property, const std::string& value)
  {
    (void)property;
    (void)value;

    if (!this->window_ || !this->window_->isInitialized())
      return;

    QMutexLocker<QRecursiveMutex> lock(&this->window_->getMutex());

    // Re-fetch the full row set from the Runner (e.g. sample counts may have
    // been updated once the dataset is loaded).
    this->window_->setModelInfoRows(this->runner->buildPredictModelInfoRows());
    this->window_->refreshModelInfoContent();
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::onLogMessage(const std::string& message, bool isError)
  {
    // When the TUI is not active, fall back to console output.
    if (!this->window_ || !this->window_->isInitialized()) {
      if (isError)
        std::cerr << "[ERROR] " << message << "\n";
      else
        std::cout << message << "\n";

      return;
    }

    // With an active TUI, log messages are informational only — progress and
    // results are shown via dedicated window methods.
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::onTimingUpdated(const std::string& metric, float value)
  {
    (void)metric;
    (void)value;

    if (!this->window_ || !this->window_->isInitialized()) {
      std::cout << "  " << metric << ": " << std::fixed << std::setprecision(2) << value << " ms\n";
      return;
    }

    // Timing updates are handled through the window's dedicated timing panel
    // when the TUI is active.
  }

  //===================================================================================================================//
  //-- Private — model info population --//
  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::populateModelInfo()
  {
    if (!this->window_ || !this->window_->isInitialized())
      return;

    if (!this->runner)
      return;

    QMutexLocker<QRecursiveMutex> lock(&this->window_->getMutex());

    this->window_->setModelInfoRows(this->runner->buildPredictModelInfoRows());
    this->window_->refreshModelInfoContent();
  }

  //===================================================================================================================//
  //-- Private — training metadata population --//
  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::populateTrainMeta()
  {
    if (!this->window_ || !this->window_->isInitialized())
      return;

    if (!this->runner)
      return;

    QMutexLocker<QRecursiveMutex> lock(&this->window_->getMutex());

    const auto& config = this->runner->getCoreConfig();
    const auto& tm = config.loadedTrainMetadata;

    std::vector<std::string> lines;

    lines.push_back(::formatPadded("Duration:", 22) + tm.durationFormatted);
    lines.push_back(::formatPadded("Samples:", 22) + std::to_string(tm.numSamples));
    lines.push_back(::formatPadded("Final Loss:", 22) + std::to_string(tm.finalLoss));
    lines.push_back(::formatPadded("Best Epoch:", 22) + std::to_string(tm.bestEpoch));
    lines.push_back(::formatPadded("Best Loss:", 22) + std::to_string(tm.bestLoss));
    lines.push_back(::formatPadded("Stop Reason:", 22) + tm.stopReason);
    lines.push_back(::formatPadded("Epochs Trained:", 22) + std::to_string(tm.lastEpoch));

    this->window_->setEpochHistoryLines(lines);
    this->window_->refreshEpochHistoryContent();
  }

  //===================================================================================================================//
  //-- Private — abort check --//
  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::checkAbortRequested()
  {
    if (this->window_ && this->window_->abortRequested() && !this->abortHandled) {
      this->abortHandled = true;
      this->runner->requestAbort();
      this->window_->updateProgressSubLine("Prediction aborted by user.");
    }
  }

  //===================================================================================================================//
  //-- Private — progress seed --//
  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::populateProgress()
  {
    if (!this->window_ || !this->window_->isInitialized())
      return;

    QMutexLocker<QRecursiveMutex> lock(&this->window_->getMutex());

    this->window_->updateProgress("Predicting 0/0", 0.0f);
  }

  //===================================================================================================================//
  //-- Explicit template instantiations --//
  //===================================================================================================================//

  template class PredictController<ANNRunner>;
  template class PredictController<CNNRunner>;

} // namespace NN_CLI
