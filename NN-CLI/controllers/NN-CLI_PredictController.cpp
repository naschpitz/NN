#include "NN-CLI_PredictController.hpp"

#include <QCoreApplication>
#include <QFutureWatcher>
#include <QTimer>
#include <QtConcurrent>

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

  PredictController::~PredictController()
  {
    // Signal connections auto-disconnect when the Runner (sender) is destroyed.
  }

  //===================================================================================================================//
  //-- Lifecycle --//
  //===================================================================================================================//

  void PredictController::init(std::unique_ptr<RunnerBase> runner)
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
        QObject::connect(this->runner.get(), &RunnerBase::sampleLoadProgress, this,
                         &PredictController::onSampleLoadProgress);
        QObject::connect(this->runner.get(), &RunnerBase::validationProgress, this,
                         &PredictController::onValidationProgress);
        QObject::connect(this->runner.get(), &RunnerBase::batchProgress, this, &PredictController::onBatchProgress);
        QObject::connect(this->runner.get(), &RunnerBase::epochCompleted, this, &PredictController::onEpochCompleted);
        QObject::connect(this->runner.get(), &RunnerBase::trainFinished, this, &PredictController::onTrainFinished);
        QObject::connect(this->runner.get(), &RunnerBase::predictFinished, this, &PredictController::onPredictFinished);
        QObject::connect(this->runner.get(), &RunnerBase::modelInfoUpdated, this,
                         &PredictController::onModelInfoUpdated);
        QObject::connect(this->runner.get(), &RunnerBase::logMessage, this, &PredictController::onLogMessage);
        QObject::connect(this->runner.get(), &RunnerBase::timingUpdated, this, &PredictController::onTimingUpdated);
      }

      this->window->startUiTimer();
    }
  }

  //===================================================================================================================//

  int PredictController::startPredict()
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
    QObject::connect(this->workWatcher.get(), &QFutureWatcher<int>::finished, this, [this]() {
      this->workResult = this->workWatcher->result();
      this->workComplete.store(true);
    });

    // Poll for completion + dismiss.  Exits the event loop once prediction is
    // done AND the user has requested dismiss/abort (pressing 'q' sets the
    // window's dismissed flag, which also serves as the abort signal).
    this->completionTimer = std::make_unique<QTimer>();
    this->completionTimer->setInterval(50);
    QObject::connect(this->completionTimer.get(), &QTimer::timeout, this, [this]() {
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

  void PredictController::shutdown()
  {
    if (this->runner)
      this->runner->disconnect(this);

    this->window.reset();
  }

  //===================================================================================================================//
  //-- Accessors --//
  //===================================================================================================================//

  RunnerBase* PredictController::getRunner() const
  {
    return this->runner.get();
  }

  //===================================================================================================================//

  TerminalUI_PredictWindow* PredictController::getWindow() const
  {
    return this->window.get();
  }

  //===================================================================================================================//
  //-- Runner signal handlers --//
  //===================================================================================================================//

  void PredictController::onSampleLoadProgress(ulong current, ulong total, ulong batchIndex, ulong totalBatches,
                                               bool isValidation)
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

  void PredictController::onValidationProgress(ulong current, ulong total)
  {
    if (!this->window || !this->window->isInitialized())
      return;
    float fraction = (total > 0) ? static_cast<float>(current) / static_cast<float>(total) : 0.0f;
    this->window->updateProgress("Validating", fraction);
  }

  //===================================================================================================================//

  void PredictController::onBatchProgress(int batchIdx, int totalBatches, float currentLoss, float samplesPerSec,
                                          float etaSeconds, const std::vector<float>& fractions)
  {
    (void)currentLoss;
    (void)samplesPerSec;
    (void)etaSeconds;

    if (!this->window || !this->window->isInitialized())
      return;

    this->checkAbortRequested();

    float fraction = fractions.empty() ? 0.0f : fractions[0];
    this->window->updateProgress("Predicting " + std::to_string(batchIdx + 1) + "/" + std::to_string(totalBatches),
                                 fraction);
    this->window->updateProgressSubLine(std::to_string(batchIdx + 1) + "/" + std::to_string(totalBatches) + " (" +
                                        std::to_string(static_cast<int>(fraction * 100)) + "%)");
  }

  //===================================================================================================================//

  void PredictController::onEpochCompleted(int epochIdx, int totalEpochs, float epochLoss, bool hasValLoss,
                                           float valLoss, float learningRate, const std::string& summary)
  {
    (void)epochIdx;
    (void)totalEpochs;
    (void)epochLoss;
    (void)hasValLoss;
    (void)valLoss;
    (void)learningRate;

    if (!this->window || !this->window->isInitialized())
      return;

    // When the TUI is active these events are informational only.
    this->window->updateProgressSubLine(summary);
  }

  //===================================================================================================================//

  void PredictController::onTrainFinished(bool success, const std::string& finalSummary)
  {
    if (!this->window || !this->window->isInitialized())
      return;

    std::string prefix = success ? "[Predict complete] " : "[Predict failed] ";
    this->window->updateProgressSubLine(prefix + finalSummary);
  }

  //===================================================================================================================//

  void PredictController::onPredictFinished(const Common::PredictResults<float>& results, size_t numInputs,
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

  void PredictController::onModelInfoUpdated(const std::string& property, const std::string& value)
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

  void PredictController::onLogMessage(const std::string& message, bool isError)
  {
    if (!this->window || !this->window->isInitialized())
      return;

    // With an active TUI, log messages are informational only — progress and
    // results are shown via dedicated window methods.
  }

  //===================================================================================================================//

  void PredictController::onTimingUpdated(const std::string& metric, float value)
  {
    (void)metric;
    (void)value;

    if (!this->window || !this->window->isInitialized())
      return;

    // Timing updates are handled through the window's dedicated timing panel
    // when the TUI is active.
  }

  //===================================================================================================================//
  //-- Private — model info population --//
  //===================================================================================================================//

  void PredictController::populateModelInfo()
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

  void PredictController::populateTrainMeta()
  {
    if (!this->window || !this->window->isInitialized())
      return;

    if (!this->runner)
      return;
    const auto& tm = this->runner->getLoadedTrainMetadata();

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

  void PredictController::checkAbortRequested()
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

  void PredictController::populateProgress()
  {
    if (!this->window || !this->window->isInitialized())
      return;
    this->window->updateProgress("Predicting 0/0", 0.0f);
  }

  //===================================================================================================================//
  //-- Explicit template instantiations --//
  //===================================================================================================================//

} // namespace NN_CLI
