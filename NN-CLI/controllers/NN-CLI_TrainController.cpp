#include "NN-CLI_TrainController.hpp"

#include "NN-CLI_ANNRunner.hpp"
#include "NN-CLI_CNNRunner.hpp"
#include "NN-CLI_TerminalUI_TrainWindow.hpp"
#include "NN-CLI_LossReferenceTable.hpp"
#include "NN-CLI_SummaryTable.hpp"

#include "Common/Common_Utils.hpp"

#include <QCoreApplication>
#include <QFutureWatcher>
#include <QTimer>
#include <QtConcurrent>

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace
{
  //===================================================================================================================//
  // Format an ETA in seconds as "mm:ss" (or "h:mm:ss" once it exceeds an hour).
  std::string formatEta(double seconds)
  {
    if (seconds < 0.0)
      seconds = 0.0;

    long total = static_cast<long>(seconds + 0.5);
    long h = total / 3600;
    long m = (total % 3600) / 60;
    long s = total % 60;

    std::ostringstream out;

    if (h > 0)
      out << h << ":" << std::setfill('0') << std::setw(2) << m << ":" << std::setw(2) << s;
    else
      out << std::setfill('0') << std::setw(2) << m << ":" << std::setw(2) << s;

    return out.str();
  }

} // namespace

namespace NN_CLI
{

  //===================================================================================================================//
  //-- Ctors / Dtors --//
  //===================================================================================================================//

  template <typename RunnerT>
  TrainController<RunnerT>::~TrainController()
  {
    // Signal connections auto-disconnect when runnerSignals (sender) is destroyed
    // (Runner owns runnerSignals; Controller owns Runner via unique_ptr).
  }

  //===================================================================================================================//
  //-- Lifecycle --//
  //===================================================================================================================//

  template <typename RunnerT>
  void TrainController<RunnerT>::init(std::unique_ptr<RunnerT> runner)
  {
    this->window = std::make_unique<TerminalUI_TrainWindow>();
    this->runner = std::move(runner);

    if (this->runner)
      this->totalEpochs = this->runner->getTotalEpochs();

    // Initialize the ncurses TUI.  If init fails (e.g. no TTY attached),
    // the window gracefully degrades -- the UI timer is never started, so
    // the training proceeds with console-only output from the Runner.
    if (this->window)
      this->window->init();

    // Populate the Model Info panel with static core configuration data.
    this->populateModelInfo();

    // Seed the training progress bar so it shows "Epoch    1/N" at 0% with
    // an empty stats line from the start, before the first batch-progress
    // event arrives.
    if (this->window) {
      this->window->updateProgress(this->buildEpochLabel(), 0.0f);
      this->window->updateProgressSubLine("Loss: 0.000000");
    }

    // Connect Runner signals only when the TUI is active.  In the no-TUI
    // path the main thread blocks synchronously in runner->train(), so
    // queued cross-thread events would pile up unbounded.
    if (this->runner && this->window && this->window->isInitialized()) {
      auto& hub = this->runner->getRunnerSignals();
      auto* ctx = &this->signalContext;

      QObject::connect(&hub, &RunnerSignals::sampleLoadProgress, ctx,
                       [this](ulong current, ulong total, ulong batchIndex, ulong totalBatches, bool isValidation) {
                         this->onSampleLoadProgress(current, total, batchIndex, totalBatches, isValidation);
                       });

      QObject::connect(&hub, &RunnerSignals::validationProgress, ctx,
                       [this](ulong current, ulong total) { this->onValidationProgress(current, total); });

      QObject::connect(&hub, &RunnerSignals::batchProgress, ctx,
                       [this](int batchIdx, int totalBatches, float currentLoss, float samplesPerSec, float etaSeconds,
                              const std::vector<float>& fractions) {
                         this->onBatchProgress(batchIdx, totalBatches, currentLoss, samplesPerSec, etaSeconds,
                                               fractions);
                       });

      QObject::connect(&hub, &RunnerSignals::epochCompleted, ctx,
                       [this](int epochIdx, int totalEpochs, float epochLoss, bool hasValLoss, float valLoss,
                              float learningRate, const std::string& summary) {
                         this->onEpochCompleted(epochIdx, totalEpochs, epochLoss, hasValLoss, valLoss, learningRate,
                                                summary);
                       });

      QObject::connect(&hub, &RunnerSignals::trainFinished, ctx, [this](bool success, const std::string& finalSummary) {
        this->onTrainFinished(success, finalSummary);
      });

      QObject::connect(
        &hub, &RunnerSignals::modelInfoUpdated, ctx,
        [this](const std::string& property, const std::string& value) { this->onModelInfoUpdated(property, value); });

      QObject::connect(&hub, &RunnerSignals::logMessage, ctx,
                       [this](const std::string& message, bool isError) { this->onLogMessage(message, isError); });

      QObject::connect(&hub, &RunnerSignals::timingUpdated, ctx,
                       [this](const std::string& metric, float value) { this->onTimingUpdated(metric, value); });
    }

    // Start the window's UI timer.  It fires once the QCoreApplication event
    // loop is entered in startTrain(); observer callbacks arrive on the same
    // main thread via queued signal delivery, so no mutex is needed.
    if (this->window)
      this->window->startUiTimer();
  }

  //===================================================================================================================//

  template <typename RunnerT>
  int TrainController<RunnerT>::startTrain()
  {
    if (!this->runner)
      return 1;

    // No TUI → training runs synchronously on the calling thread (same as
    // pre-Phase-2 behavior; no event loop is started).
    if (!this->window || !this->window->isInitialized())
      return this->runner->train();

    // TUI → training runs on a QtConcurrent worker thread while the main
    // thread spins a QCoreApplication event loop to drive the UI timer.
    this->workComplete.store(false);
    this->workResult = 0;

    this->workWatcher = std::make_unique<QFutureWatcher<int>>();
    QObject::connect(this->workWatcher.get(), &QFutureWatcher<int>::finished, &this->signalContext, [this]() {
      this->workResult = this->workWatcher->result();
      this->workComplete.store(true);
    });

    // Poll for completion + dismiss.  Exits the event loop once training is
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

    this->workWatcher->setFuture(QtConcurrent::run([this]() { return this->runner->train(); }));

    return QCoreApplication::exec();
  }

  //===================================================================================================================//
  //-- Accessors --//
  //===================================================================================================================//

  template <typename RunnerT>
  TerminalUI_TrainWindow* TrainController<RunnerT>::getWindow() const
  {
    return this->window.get();
  }

  //===================================================================================================================//

  template <typename RunnerT>
  RunnerT* TrainController<RunnerT>::getRunner() const
  {
    return this->runner.get();
  }

  //===================================================================================================================//
  //-- Runner signal handlers --//
  //===================================================================================================================//

  template <typename RunnerT>
  void TrainController<RunnerT>::onSampleLoadProgress(ulong current, ulong total, ulong batchIndex, ulong totalBatches,
                                                      bool isValidation)
  {
    (void)batchIndex;
    (void)totalBatches;
    (void)isValidation;

    if (!this->window)
      return;
    float fraction = (total > 0) ? static_cast<float>(current) / static_cast<float>(total) : 0.0f;

    std::string label =
      "Samples " + SummaryTable::formatWithCommas(current) + "/" + SummaryTable::formatWithCommas(total);

    this->window->setLoadingProgress(label, fraction);
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TrainController<RunnerT>::onValidationProgress(ulong current, ulong total)
  {
    if (!this->window)
      return;
    this->isValidating = true;

    float fraction = (total > 0) ? static_cast<float>(current) / static_cast<float>(total) : 0.0f;
    this->window->updateProgress("Validating", fraction);
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TrainController<RunnerT>::onBatchProgress(int batchIdx, int totalBatches, float currentLoss, float samplesPerSec,
                                                 float etaSeconds, const std::vector<float>& fractions)
  {
    (void)batchIdx;
    (void)totalBatches;

    if (!this->window)
      return;
    this->checkAbortRequested();

    // Clear any transitional "Validating" state from the previous epoch.
    this->isValidating = false;

    this->window->updateProgress(this->buildEpochLabel(), fractions);

    // Sub-line: running average loss, current learning rate, ingestion rate, and epoch ETA.
    std::ostringstream stats;
    stats << "Loss: " << std::fixed << std::setprecision(6) << currentLoss << " | LR " << std::defaultfloat
          << std::setprecision(6) << this->runner->getCurrentLearningRate() << " | " << std::fixed
          << std::setprecision(1) << samplesPerSec << " img/s | ETA " << formatEta(etaSeconds);
    this->window->updateProgressSubLine(stats.str());

    this->refreshTimingPanel();
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TrainController<RunnerT>::onEpochCompleted(int epochIdx, int totalEpochs, float epochLoss, bool hasValLoss,
                                                  float valLoss, float learningRate, const std::string& summary)
  {
    if (!this->window)
      return;
    this->checkAbortRequested();

    // Track the current epoch (0-based index → next epoch number for display).
    this->currentEpoch = epochIdx + 1;

    // Build a structured table row for the epoch data.
    std::string epochStr = std::to_string(epochIdx + 1);

    std::ostringstream lossStream;
    lossStream << std::fixed << std::setprecision(6) << epochLoss;

    std::string valLossStr;

    if (hasValLoss) {
      std::ostringstream valLossStream;
      valLossStream << std::fixed << std::setprecision(6) << valLoss;
      valLossStr = valLossStream.str();
    } else {
      valLossStr = "-";
    }

    std::ostringstream learningRateStream;
    learningRateStream << std::defaultfloat << std::setprecision(6) << learningRate;

    bool isBest = summary.find("Best*") != std::string::npos;
    std::string bestStr = isBest ? "✓" : "";
    std::string timestamp = Common::Utils::formatHumanReadable();

    TerminalUI_Table::Row row = {epochStr, lossStream.str(), valLossStr, learningRateStream.str(), bestStr, timestamp};

    this->window->addEpochRow(row);
    this->window->refreshEpochContent();

    // When validation was performed for this epoch, show a transitional
    // "Validating" progress bar that persists until the next onBatchProgress
    // event replaces it with "Training".  The "Validating" bar replaces the
    // progress bar content, NOT the panel title.
    if (hasValLoss) {
      this->isValidating = true;
      this->window->updateProgress("Validating", 1.0f);
    }
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TrainController<RunnerT>::onTrainFinished(bool success, const std::string& finalSummary)
  {
    if (!this->window)
      return;
    this->isValidating = false;

    std::string prefix = success ? "[Training complete] " : "[Training failed] ";
    this->window->addLogMessage(prefix + finalSummary);
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TrainController<RunnerT>::onModelInfoUpdated(const std::string& property, const std::string& value)
  {
    (void)property;
    (void)value;

    if (!this->window || !this->runner)
      return;
    // The Runner emits these notifications after it has updated its internal
    // state (e.g. sample counts once the dataset is loaded), so rebuild the
    // whole configuration section from the authoritative row set rather than
    // appending one raw key/value at a time.
    this->window->setModelInfoRows(this->runner->buildModelInfoRows());
    this->window->refreshModelInfoContent();
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TrainController<RunnerT>::onLogMessage(const std::string& message, bool isError)
  {
    if (!this->window)
      return;
    std::string formatted = isError ? ("[ERROR] " + message) : message;
    this->window->addLogMessage(formatted);
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TrainController<RunnerT>::onTimingUpdated(const std::string& metric, float value)
  {
    (void)metric;
    (void)value;

    if (!this->window)
      return;
    this->refreshTimingPanel();
  }

  //===================================================================================================================//
  //-- Private — epoch label --//
  //===================================================================================================================//

  template <typename RunnerT>
  std::string TrainController<RunnerT>::buildEpochLabel() const
  {
    std::ostringstream oss;
    oss << "Epoch " << std::setw(4) << (this->currentEpoch + 1) << "/" << this->totalEpochs;

    return oss.str();
  }

  //===================================================================================================================//
  //-- Private — timing panel refresh --//
  //===================================================================================================================//

  template <typename RunnerT>
  void TrainController<RunnerT>::refreshTimingPanel()
  {
    if (!this->window || !this->runner)
      return;

    int width = this->window->getTimingPanel()->contentWidth();
    std::vector<std::string> lines = this->runner->getTimingLines(width);

    if (lines.empty())
      return;

    this->window->setTimingLines(lines);
    this->window->refreshTimingContent();
  }

  //===================================================================================================================//
  //-- Private — check abort --//
  //===================================================================================================================//

  template <typename RunnerT>
  void TrainController<RunnerT>::checkAbortRequested()
  {
    if (this->window && this->window->abortRequested() && !this->abortHandled) {
      this->abortHandled = true;
      this->runner->requestAbort();
      this->window->addLogMessage("Training aborted by user.");
    }
  }

  //===================================================================================================================//
  //-- Private — model info population --//
  //===================================================================================================================//

  template <typename RunnerT>
  void TrainController<RunnerT>::populateModelInfo()
  {
    if (!this->window || !this->runner)
      return;

    //-- Model configuration section --//
    // Mirror the pre-refactoring Summary table exactly: the Runner builds the
    // full set of rows (device, layers, parameters, sample counts, training
    // hyper-parameters, cost function, ...) including section separators.
    // Sample counts are zero until train() loads the data; onModelInfoUpdated()
    // re-fetches these rows once the counts are known.
    this->window->setModelInfoTitle("Model Configuration");
    this->window->setModelInfoRows(this->runner->buildModelInfoRows());

    //-- Loss Reference section --//
    ulong numClasses = this->runner->getNumOutputClasses();
    this->window->setLossReferenceRows(LossReferenceTable::collectRows(numClasses));

    this->window->refreshModelInfoContent();
  }

  //===================================================================================================================//
  //-- Explicit template instantiations --//
  //===================================================================================================================//

  template class TrainController<ANNRunner>;
  template class TrainController<CNNRunner>;

} // namespace NN_CLI
