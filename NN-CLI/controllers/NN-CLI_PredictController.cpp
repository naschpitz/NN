#include "NN-CLI_PredictController.hpp"

#include "NN-CLI_ANNRunner.hpp"
#include "NN-CLI_CNNRunner.hpp"

#include <QCoreApplication>
#include <QFutureWatcher>
#include <QTimer>
#include <QtConcurrent>

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
    // Signal connections auto-disconnect when runnerSignals (sender) is destroyed.
  }

  //===================================================================================================================//
  //-- Lifecycle --//
  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::init(std::unique_ptr<RunnerT> runner)
  {
    this->window = std::make_unique<TerminalUI_PredictWindow>();
    this->runner = std::move(runner);

    // Initialize the ncurses TUI.  If init fails (e.g. no TTY attached),
    // the window gracefully degrades — the UI timer is never started, so
    // the prediction proceeds with console-only output from the Runner.
    if (this->window && this->window->init()) {
      this->populateModelInfo();
      this->populateTrainMeta();
      this->populateProgress();

      // Connect Runner signals only when the TUI is active.  In the no-TUI
      // path the main thread blocks synchronously in runner->predict(), so
      // queued cross-thread events would pile up unbounded.
      if (this->runner) {
        auto& hub = this->runner->getRunnerSignals();
        auto* ctx = &this->signalContext;

        QObject::connect(&hub, &RunnerSignals::sampleLoadProgress, ctx,
                         [this](ulong current, ulong total, ulong batchIndex, ulong totalBatches, bool isValidation) {
                           this->onSampleLoadProgress(current, total, batchIndex, totalBatches, isValidation);
                         });

        QObject::connect(&hub, &RunnerSignals::validationProgress, ctx,
                         [this](ulong current, ulong total) { this->onValidationProgress(current, total); });

        QObject::connect(&hub, &RunnerSignals::batchProgress, ctx,
                         [this](int batchIdx, int totalBatches, float currentLoss, float samplesPerSec,
                                float etaSeconds, const std::vector<float>& fractions) {
                           this->onBatchProgress(batchIdx, totalBatches, currentLoss, samplesPerSec, etaSeconds,
                                                 fractions);
                         });

        QObject::connect(&hub, &RunnerSignals::epochCompleted, ctx,
                         [this](int epochIdx, int totalEpochs, float epochLoss, bool hasValLoss, float valLoss,
                                float learningRate, const std::string& summary) {
                           this->onEpochCompleted(epochIdx, totalEpochs, epochLoss, hasValLoss, valLoss, learningRate,
                                                  summary);
                         });

        QObject::connect(
          &hub, &RunnerSignals::trainFinished, ctx,
          [this](bool success, const std::string& finalSummary) { this->onTrainFinished(success, finalSummary); });

        QObject::connect(&hub, &RunnerSignals::predictFinished, ctx,
                         [this](const Common::PredictResults<float>& results, size_t numInputs, double durationSeconds,
                                const std::string& durationFormatted, const std::string& outputPath) {
                           this->onPredictFinished(results, numInputs, durationSeconds, durationFormatted, outputPath);
                         });

        QObject::connect(
          &hub, &RunnerSignals::modelInfoUpdated, ctx,
          [this](const std::string& property, const std::string& value) { this->onModelInfoUpdated(property, value); });

        QObject::connect(&hub, &RunnerSignals::logMessage, ctx,
                         [this](const std::string& message, bool isError) { this->onLogMessage(message, isError); });

        QObject::connect(&hub, &RunnerSignals::timingUpdated, ctx,
                         [this](const std::string& metric, float value) { this->onTimingUpdated(metric, value); });
      }

      this->window->startUiTimer();
    }
  }

  //===================================================================================================================//

  template <typename RunnerT>
  int PredictController<RunnerT>::startPredict()
  {
    if (!this->runner)
      return 1;

    // No TUI → prediction runs synchronously on the calling thread (same as
    // pre-Phase-2 behavior; no event loop is started).
    if (!this->window || !this->window->isInitialized())
      return this->runner->predict();

    // TUI → prediction runs on a QtConcurrent worker thread while the main
    // thread spins a QCoreApplication event loop to drive the UI timer.
    this->workComplete.store(false);
    this->workResult = 0;

    this->workWatcher = std::make_unique<QFutureWatcher<int>>();
    QObject::connect(this->workWatcher.get(), &QFutureWatcher<int>::finished, &this->signalContext, [this]() {
      this->workResult = this->workWatcher->result();
      this->workComplete.store(true);
    });

    // Poll for completion + dismiss.  Exits the event loop once prediction is
    // done AND the user has requested dismiss/abort (pressing 'q' sets the
    // window's dismissed flag, which also serves as the abort signal).
    this->completionTimer = std::make_unique<QTimer>();
    this->completionTimer->setInterval(50);
    QObject::connect(this->completionTimer.get(), &QTimer::timeout, &this->signalContext, [this]() {
      if (this->workComplete.load() && this->window && this->window->abortRequested()) {
        this->completionTimer->stop();
        QCoreApplication::exit(this->workResult);
      }
    });

    this->completionTimer->start();

    this->workWatcher->setFuture(QtConcurrent::run([this]() { return this->runner->predict(); }));

    return QCoreApplication::exec();
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::shutdown()
  {
    if (this->runner)
      this->runner->getRunnerSignals().disconnect(&this->signalContext);

    this->window.reset();
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
    return this->window.get();
  }

  //===================================================================================================================//
  //-- Runner signal handlers --//
  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::onSampleLoadProgress(ulong current, ulong total, ulong batchIndex,
                                                        ulong totalBatches, bool isValidation)
  {
    (void)batchIndex;
    (void)totalBatches;
    (void)isValidation;

    if (!this->window || !this->window->isInitialized())
      return;
    float fraction = (total > 0) ? static_cast<float>(current) / static_cast<float>(total) : 0.0f;

    this->window->setLoadingProgress("Samples " + std::to_string(current) + "/" + std::to_string(total), fraction);
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::onValidationProgress(ulong current, ulong total)
  {
    if (!this->window || !this->window->isInitialized())
      return;
    float fraction = (total > 0) ? static_cast<float>(current) / static_cast<float>(total) : 0.0f;
    this->window->updateProgress("Validating", fraction);
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

    if (!this->window || !this->window->isInitialized()) {
      // Console fallback.
      float fraction = fractions.empty() ? 0.0f : fractions[0];
      std::cout << "\r  Progress: " << (batchIdx + 1) << "/" << totalBatches << " (" << std::fixed
                << std::setprecision(1) << (fraction * 100.0f) << "%)" << std::flush;
      return;
    }

    this->checkAbortRequested();

    float fraction = fractions.empty() ? 0.0f : fractions[0];
    this->window->updateProgress("Predicting " + std::to_string(batchIdx + 1) + "/" + std::to_string(totalBatches),
                                 fraction);
    this->window->updateProgressSubLine(std::to_string(batchIdx + 1) + "/" + std::to_string(totalBatches) + " (" +
                                        std::to_string(static_cast<int>(fraction * 100)) + "%)");
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::onEpochCompleted(int epochIdx, int totalEpochs, float epochLoss, bool hasValLoss,
                                                    float valLoss, float learningRate, const std::string& summary)
  {
    (void)epochIdx;
    (void)totalEpochs;
    (void)epochLoss;
    (void)hasValLoss;
    (void)valLoss;
    (void)learningRate;

    // When the TUI is not active, print to console for interface completeness.
    if (!this->window || !this->window->isInitialized()) {
      std::cout << summary << "\n";
      return;
    }

    // When the TUI is active these events are informational only.
    this->window->updateProgressSubLine(summary);
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::onTrainFinished(bool success, const std::string& finalSummary)
  {
    if (!this->window || !this->window->isInitialized()) {
      std::string prefix = success ? "[Predict complete] " : "[Predict failed] ";
      std::cout << "\n" << prefix << finalSummary << "\n";
      return;
    }

    std::string prefix = success ? "[Predict complete] " : "[Predict failed] ";
    this->window->updateProgressSubLine(prefix + finalSummary);
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::onPredictFinished(const Common::PredictResults<float>& results, size_t numInputs,
                                                     double durationSeconds, const std::string& durationFormatted,
                                                     const std::string& outputPath)
  {
    (void)durationSeconds;

    if (!this->window || !this->window->isInitialized())
      return;
    const auto& ioConfig = this->runner->getIOConfig();

    // Image output: single summary line.
    if (ioConfig.outputType == DataType::IMAGE) {
      this->window->addResultRow({"", std::to_string(numInputs) + " image outputs -> " + outputPath, ""});
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

        this->window->addResultRow({std::to_string(i), classStr, pctStr});
      }
    }

    this->window->refreshResultsContent();
    this->window->updateProgress("Predicting", 1.0f);
    this->window->updateProgressSubLine("Done — " + durationFormatted + ", output: " + outputPath);
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::onModelInfoUpdated(const std::string& property, const std::string& value)
  {
    (void)property;
    (void)value;

    if (!this->window || !this->window->isInitialized())
      return;
    // Re-fetch the full row set from the Runner (e.g. sample counts may have
    // been updated once the dataset is loaded).
    this->window->setModelInfoRows(this->runner->buildPredictModelInfoRows());
    this->window->refreshModelInfoContent();
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::onLogMessage(const std::string& message, bool isError)
  {
    // When the TUI is not active, fall back to console output.
    if (!this->window || !this->window->isInitialized()) {
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

    if (!this->window || !this->window->isInitialized()) {
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
    if (!this->window || !this->window->isInitialized())
      return;

    if (!this->runner)
      return;
    this->window->setModelInfoRows(this->runner->buildPredictModelInfoRows());
    this->window->refreshModelInfoContent();
  }

  //===================================================================================================================//
  //-- Private — training metadata population --//
  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::populateTrainMeta()
  {
    if (!this->window || !this->window->isInitialized())
      return;

    if (!this->runner)
      return;
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

    this->window->setEpochHistoryLines(lines);
    this->window->refreshEpochHistoryContent();
  }

  //===================================================================================================================//
  //-- Private — abort check --//
  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::checkAbortRequested()
  {
    if (this->window && this->window->abortRequested() && !this->abortHandled) {
      this->abortHandled = true;
      this->runner->requestAbort();
      this->window->updateProgressSubLine("Prediction aborted by user.");
    }
  }

  //===================================================================================================================//
  //-- Private — progress seed --//
  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::populateProgress()
  {
    if (!this->window || !this->window->isInitialized())
      return;
    this->window->updateProgress("Predicting 0/0", 0.0f);
  }

  //===================================================================================================================//
  //-- Explicit template instantiations --//
  //===================================================================================================================//

  template class PredictController<ANNRunner>;
  template class PredictController<CNNRunner>;

} // namespace NN_CLI
