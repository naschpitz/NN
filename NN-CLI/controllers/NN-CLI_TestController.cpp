#include "NN-CLI_TestController.hpp"

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
  std::string formatPercent(float fraction)
  {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << (fraction * 100.0f) << "%";
    return oss.str();
  }

} // namespace

namespace NN_CLI
{

  //===================================================================================================================//
  //-- Ctors / Dtors --//
  //===================================================================================================================//

  TestController::~TestController()
  {
    // Signal connections auto-disconnect when the Runner (sender) is destroyed.
  }

  //===================================================================================================================//
  //-- Lifecycle --//
  //===================================================================================================================//

  void TestController::init(std::unique_ptr<RunnerBase> runner)
  {
    this->window = std::make_unique<TerminalUI_TestWindow>();
    this->runner = std::move(runner);

    // Initialize the ncurses TUI.  If init fails (e.g. no TTY attached),
    // the window gracefully degrades — the UI timer is never started, so
    // the test proceeds with console-only output from the Runner.
    if (this->window && this->window->init()) {
      this->populateModelInfo();
      this->populateTrainMeta();
      this->populateProgress();

      // Connect Runner signals only when the TUI is active.  In the no-TUI
      // path the main thread blocks synchronously in runner->test(), so
      // queued cross-thread events would pile up unbounded.
      if (this->runner) {
        QObject::connect(this->runner.get(), &RunnerBase::sampleLoadProgress, this,
                         &TestController::onSampleLoadProgress);
        QObject::connect(this->runner.get(), &RunnerBase::batchProgress, this, &TestController::onBatchProgress);
        QObject::connect(this->runner.get(), &RunnerBase::testFinished, this, &TestController::onTestFinished);
        QObject::connect(this->runner.get(), &RunnerBase::modelInfoUpdated, this, &TestController::onModelInfoUpdated);
        QObject::connect(this->runner.get(), &RunnerBase::logMessage, this, &TestController::onLogMessage);
        QObject::connect(this->runner.get(), &RunnerBase::timingUpdated, this, &TestController::onTimingUpdated);
      }

      this->window->startUiTimer();
    }
  }

  //===================================================================================================================//

  int TestController::startTest()
  {
    if (!this->runner)
      return 1;

    // No TUI → test runs synchronously on the calling thread (same as the
    // pre-TUI behavior; no event loop is started).  The Runner owns console
    // output in this headless path.
    if (!this->window || !this->window->isInitialized())
      return this->runner->test();

    // TUI → test runs on a QtConcurrent worker thread while the main thread
    // spins a QCoreApplication event loop to drive the UI timer.
    this->workComplete.store(false);
    this->workResult = 0;

    this->workWatcher = std::make_unique<QFutureWatcher<int>>();
    QObject::connect(this->workWatcher.get(), &QFutureWatcher<int>::finished, this, [this]() {
      this->workResult = this->workWatcher->result();
      this->workComplete.store(true);
    });

    // Poll for completion + dismiss.  Exits the event loop once test is done
    // AND the user has requested dismiss/abort (pressing 'q' sets the window's
    // dismissed flag, which also serves as the abort signal).
    this->completionTimer = std::make_unique<QTimer>();
    this->completionTimer->setInterval(50);
    QObject::connect(this->completionTimer.get(), &QTimer::timeout, this, [this]() {
      if (this->workComplete.load() && this->window && this->window->abortRequested()) {
        this->completionTimer->stop();
        QCoreApplication::exit(this->workResult);
      }
    });

    this->completionTimer->start();

    this->workWatcher->setFuture(QtConcurrent::run([this]() { return this->runner->test(); }));

    return QCoreApplication::exec();
  }

  //===================================================================================================================//

  void TestController::shutdown()
  {
    if (this->runner)
      this->runner->disconnect(this);

    this->window.reset();
  }

  //===================================================================================================================//
  //-- Accessors --//
  //===================================================================================================================//

  RunnerBase* TestController::getRunner() const
  {
    return this->runner.get();
  }

  //===================================================================================================================//

  TerminalUI_TestWindow* TestController::getWindow() const
  {
    return this->window.get();
  }

  //===================================================================================================================//
  //-- Runner signal handlers --//
  //===================================================================================================================//

  void TestController::onSampleLoadProgress(ulong current, ulong total, ulong batchIndex, ulong totalBatches,
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

  void TestController::onBatchProgress(int batchIdx, int totalBatches, float currentLoss, float samplesPerSec,
                                       float etaSeconds, const std::vector<float>& fractions)
  {
    (void)currentLoss;
    (void)samplesPerSec;
    (void)etaSeconds;

    if (!this->window || !this->window->isInitialized())
      return;

    this->checkAbortRequested();

    float fraction = fractions.empty() ? 0.0f : fractions[0];
    this->window->updateProgress("Testing " + std::to_string(batchIdx + 1) + "/" + std::to_string(totalBatches),
                                 fraction);
    this->window->updateProgressSubLine(std::to_string(batchIdx + 1) + "/" + std::to_string(totalBatches) + " (" +
                                        std::to_string(static_cast<int>(fraction * 100)) + "%)");
  }

  //===================================================================================================================//

  void TestController::onTestFinished(const Common::TestResult<float>& result, double durationSeconds,
                                      const std::string& durationFormatted, const std::string& outputPath)
  {
    (void)durationSeconds;

    if (!this->window || !this->window->isInitialized())
      return;

    this->window->clearResultRows();

    const auto& cm = result.confusionMatrix;

    // Per-class rows.
    for (ulong i = 0; i < cm.numClasses; ++i) {
      this->window->addResultRow({std::to_string(i), ::formatPercent(cm.precision[i]), ::formatPercent(cm.recall[i]),
                                  ::formatPercent(cm.f1Score[i]), std::to_string(cm.support[i])});
    }

    // Aggregate rows (macro / micro / weighted averages).
    this->window->addResultRow(
      {"Macro", ::formatPercent(cm.macroPrecision), ::formatPercent(cm.macroRecall), ::formatPercent(cm.macroF1), ""});
    this->window->addResultRow(
      {"Micro", ::formatPercent(cm.microPrecision), ::formatPercent(cm.microRecall), ::formatPercent(cm.microF1), ""});
    this->window->addResultRow({"Weighted", ::formatPercent(cm.weightedPrecision), ::formatPercent(cm.weightedRecall),
                                ::formatPercent(cm.weightedF1), ""});

    this->window->refreshResultsContent();

    // One-line summary for the progress subline.
    std::ostringstream summary;
    summary << std::fixed << std::setprecision(2) << "Accuracy: " << result.accuracy << "%"
            << "  Avg loss: " << std::setprecision(6) << result.averageLoss << "  " << result.numCorrect << "/"
            << result.numSamples << " correct"
            << "  " << durationFormatted << ", output: " << outputPath;

    this->window->updateProgress("Testing", 1.0f);
    this->window->updateProgressSubLine(summary.str());
  }

  //===================================================================================================================//

  void TestController::onModelInfoUpdated(const std::string& property, const std::string& value)
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

  void TestController::onLogMessage(const std::string& message, bool isError)
  {
    (void)message;
    (void)isError;

    if (!this->window || !this->window->isInitialized())
      return;

    // With an active TUI, log messages are informational only — progress and
    // results are shown via dedicated window methods.
  }

  //===================================================================================================================//

  void TestController::onTimingUpdated(const std::string& metric, float value)
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

  void TestController::populateModelInfo()
  {
    if (!this->window || !this->window->isInitialized())
      return;

    if (!this->runner)
      return;

    this->window->setModelInfoRows(this->runner->buildModelInfoRows());
    this->window->refreshModelInfoContent();
  }

  //===================================================================================================================//
  //-- Private — training metadata population --//
  //===================================================================================================================//

  void TestController::populateTrainMeta()
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

  void TestController::checkAbortRequested()
  {
    if (this->window && this->window->abortRequested() && !this->abortHandled) {
      this->abortHandled = true;
      this->runner->requestAbort();
      this->window->updateProgressSubLine("Test aborted by user.");
    }
  }

  //===================================================================================================================//
  //-- Private — progress seed --//
  //===================================================================================================================//

  void TestController::populateProgress()
  {
    if (!this->window || !this->window->isInitialized())
      return;

    this->window->updateProgress("Testing 0/0", 0.0f);
  }

  //===================================================================================================================//
  //-- Explicit template instantiations --//
  //===================================================================================================================//

} // namespace NN_CLI
