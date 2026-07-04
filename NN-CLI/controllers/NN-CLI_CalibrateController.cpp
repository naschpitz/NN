#include "NN-CLI_CalibrateController.hpp"

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

  // Right-pad a string with spaces to the given width.  If the string already
  // exceeds the width it is returned unchanged.
  std::string formatPadded(const std::string& s, int width)
  {
    if (static_cast<int>(s.size()) >= width)
      return s;

    return s + std::string(static_cast<std::string::size_type>(width - s.size()), ' ');
  }

  //===================================================================================================================//

  // Format a 0-1 fraction as a percentage string with two decimals, e.g.
  // 0.9523 -> "95.23%".
  std::string formatPercent(double fraction)
  {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << (fraction * 100.0) << "%";
    return oss.str();
  }

  //===================================================================================================================//

  // Format a value with a fixed number of decimals, e.g. 1.2345 -> "1.2345".
  std::string formatFixed(double value, int precision)
  {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
  }

} // namespace

namespace NN_CLI
{

  //===================================================================================================================//
  //-- Ctors / Dtors --//
  //===================================================================================================================//

  CalibrateController::~CalibrateController()
  {
    // Signal connections auto-disconnect when the Runner (sender) is destroyed.
  }

  //===================================================================================================================//
  //-- Lifecycle --//
  //===================================================================================================================//

  void CalibrateController::init(std::unique_ptr<RunnerBase> runner)
  {
    this->window = std::make_unique<TerminalUI_CalibrateWindow>();
    this->runner = std::move(runner);

    // Initialize the ncurses TUI.  If init fails (e.g. no TTY attached),
    // the window gracefully degrades — the UI timer is never started, so
    // calibration proceeds with console-only output from the Runner.
    if (this->window && this->window->init()) {
      this->populateModelInfo();
      this->populateTrainMeta();
      this->populateProgress();

      // Connect Runner signals only when the TUI is active.  In the no-TUI
      // path the main thread blocks synchronously in runner->calibrate(),
      // so queued cross-thread events would pile up unbounded.
      if (this->runner) {
        QObject::connect(this->runner.get(), &RunnerBase::sampleLoadProgress, this,
                         &CalibrateController::onSampleLoadProgress);
        QObject::connect(this->runner.get(), &RunnerBase::batchProgress, this, &CalibrateController::onBatchProgress);
        QObject::connect(this->runner.get(), &RunnerBase::calibrateFinished, this,
                         &CalibrateController::onCalibrateFinished);
        QObject::connect(this->runner.get(), &RunnerBase::modelInfoUpdated, this,
                         &CalibrateController::onModelInfoUpdated);
        QObject::connect(this->runner.get(), &RunnerBase::logMessage, this, &CalibrateController::onLogMessage);
        QObject::connect(this->runner.get(), &RunnerBase::timingUpdated, this, &CalibrateController::onTimingUpdated);
      }

      this->window->startUiTimer();
    }
  }

  //===================================================================================================================//

  int CalibrateController::startCalibrate()
  {
    if (!this->runner)
      return 1;

    // No TUI -> calibration runs synchronously on the calling thread (same as
    // the pre-TUI behavior; no event loop is started).  The Runner owns
    // console output in this headless path.
    if (!this->window || !this->window->isInitialized())
      return this->runner->calibrate();

    // TUI -> calibration runs on a QtConcurrent worker thread while the main
    // thread spins a QCoreApplication event loop to drive the UI timer.
    this->workComplete.store(false);
    this->workResult = 0;

    this->workWatcher = std::make_unique<QFutureWatcher<int>>();
    QObject::connect(this->workWatcher.get(), &QFutureWatcher<int>::finished, this, [this]() {
      this->workResult = this->workWatcher->result();
      this->workComplete.store(true);
    });

    // Poll for completion + dismiss.  Exits the event loop once calibration is
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

    this->workWatcher->setFuture(QtConcurrent::run([this]() { return this->runner->calibrate(); }));

    return QCoreApplication::exec();
  }

  //===================================================================================================================//

  void CalibrateController::shutdown()
  {
    if (this->runner)
      this->runner->disconnect(this);

    this->window.reset();
  }

  //===================================================================================================================//
  //-- Accessors --//
  //===================================================================================================================//

  RunnerBase* CalibrateController::getRunner() const
  {
    return this->runner.get();
  }

  //===================================================================================================================//

  TerminalUI_CalibrateWindow* CalibrateController::getWindow() const
  {
    return this->window.get();
  }

  //===================================================================================================================//
  //-- Runner signal handlers --//
  //===================================================================================================================//

  void CalibrateController::onSampleLoadProgress(ulong current, ulong total, ulong batchIndex, ulong totalBatches,
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

  void CalibrateController::onBatchProgress(int batchIdx, int totalBatches, float currentLoss, float samplesPerSec,
                                            float etaSeconds, const std::vector<float>& fractions)
  {
    (void)currentLoss;
    (void)samplesPerSec;
    (void)etaSeconds;

    if (!this->window || !this->window->isInitialized())
      return;

    this->checkAbortRequested();

    float fraction = fractions.empty() ? 0.0f : fractions[0];
    this->window->updateProgress("Calibrating " + std::to_string(batchIdx + 1) + "/" + std::to_string(totalBatches),
                                 fraction);
    this->window->updateProgressSubLine(std::to_string(batchIdx + 1) + "/" + std::to_string(totalBatches) + " (" +
                                        std::to_string(static_cast<int>(fraction * 100)) + "%)");
  }

  //===================================================================================================================//

  void CalibrateController::onCalibrateFinished(const NN_CLI::CalibrateResult& result)
  {
    if (!this->window || !this->window->isInitialized())
      return;

    this->window->clearResultRows();

    std::size_t idRejected = result.idCount - result.idAccepted;
    std::size_t oodAccepted = result.oodCount - result.oodRejected;

    this->window->addResultRow({"Free energy threshold", ::formatFixed(result.freeEnergyThreshold, 4)});
    this->window->addResultRow({"ID percentile", ::formatFixed(result.idPercentileUsed, 1)});
    this->window->addResultRow({"ID samples", std::to_string(result.idCount)});
    this->window->addResultRow({"OOD samples", std::to_string(result.oodCount)});
    this->window->addResultRow(
      {"ID accepted", std::to_string(result.idAccepted) + " (" + ::formatPercent(result.idAcceptanceRate) + ")"});
    this->window->addResultRow(
      {"ID rejected", std::to_string(idRejected) + " (" + ::formatPercent(1.0 - result.idAcceptanceRate) + ")"});
    this->window->addResultRow(
      {"OOD accepted", std::to_string(oodAccepted) + " (" + ::formatPercent(1.0 - result.oodRejectionRate) + ")"});
    this->window->addResultRow(
      {"OOD rejected", std::to_string(result.oodRejected) + " (" + ::formatPercent(result.oodRejectionRate) + ")"});

    this->window->refreshResultsContent();

    // One-line summary for the progress subline.
    this->window->updateProgress("Calibrating", 1.0f);
    this->window->updateProgressSubLine("Calibration done in " + result.durationFormatted +
                                        ", output: " + result.outputPath);
  }

  //===================================================================================================================//

  void CalibrateController::onModelInfoUpdated(const std::string& property, const std::string& value)
  {
    (void)property;
    (void)value;

    if (!this->window || !this->window->isInitialized())
      return;

    // Re-fetch the full row set from the Runner (e.g. sample counts may have
    // been updated once the dataset is loaded).
    this->window->setModelInfoRows(this->runner->buildModelInfoRows());
    this->window->refreshModelInfoContent();
  }

  //===================================================================================================================//

  void CalibrateController::onLogMessage(const std::string& message, bool isError)
  {
    (void)message;
    (void)isError;

    if (!this->window || !this->window->isInitialized())
      return;

    // With an active TUI, log messages are informational only — progress and
    // results are shown via dedicated window methods.
  }

  //===================================================================================================================//

  void CalibrateController::onTimingUpdated(const std::string& metric, float value)
  {
    (void)metric;
    (void)value;

    if (!this->window || !this->window->isInitialized())
      return;

    // Timing updates are handled through the window's dedicated timing panel
    // when the TUI is active.
  }

  //===================================================================================================================//
  //-- Private -- model info population --//
  //===================================================================================================================//

  void CalibrateController::populateModelInfo()
  {
    if (!this->window || !this->window->isInitialized())
      return;

    if (!this->runner)
      return;

    this->window->setModelInfoRows(this->runner->buildModelInfoRows());
    this->window->refreshModelInfoContent();
  }

  //===================================================================================================================//
  //-- Private -- training metadata population --//
  //===================================================================================================================//

  void CalibrateController::populateTrainMeta()
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
  //-- Private -- abort check --//
  //===================================================================================================================//

  void CalibrateController::checkAbortRequested()
  {
    if (this->window && this->window->abortRequested() && !this->abortHandled) {
      this->abortHandled = true;
      this->runner->requestAbort();
      this->window->updateProgressSubLine("Calibration aborted by user.");
    }
  }

  //===================================================================================================================//
  //-- Private -- progress seed --//
  //===================================================================================================================//

  void CalibrateController::populateProgress()
  {
    if (!this->window || !this->window->isInitialized())
      return;

    this->window->updateProgress("Calibrating 0/0", 0.0f);
  }

  //===================================================================================================================//
  //-- Explicit template instantiations --//
  //===================================================================================================================//

} // namespace NN_CLI
