#include "NN-CLI_TrainingController.hpp"

#include "NN-CLI_ANNRunner.hpp"
#include "NN-CLI_CNNRunner.hpp"
#include "NN-CLI_TerminalUI_TrainingWindow.hpp"
#include "NN-CLI_LossReferenceTable.hpp"
#include "NN-CLI_SummaryTable.hpp"

#include "Common/Common_Utils.hpp"

#include <QMutex>

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
  TrainingController<RunnerT>::~TrainingController()
  {
    if (this->runner)
      this->runner->removeObserver(this);
  }

  //===================================================================================================================//
  //-- Lifecycle --//
  //===================================================================================================================//

  template <typename RunnerT>
  void TrainingController<RunnerT>::init(std::unique_ptr<RunnerT> runner)
  {
    this->window = std::make_unique<TerminalUI_TrainingWindow>();
    this->runner = std::move(runner);

    if (this->runner) {
      this->runner->addObserver(this);
      this->totalEpochs = this->runner->getTotalEpochs();
    }

    // Initialize the ncurses TUI.  If init fails (e.g. no TTY attached),
    // the window gracefully degrades -- the UI thread is never started, so
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

    // Start the window's dedicated UI thread.  From here on the window
    // redraws itself at a fixed frame rate and handles input and resize on
    // its own thread; the observer callbacks below only update view data
    // under the window mutex.
    if (this->window)
      this->window->startUiThread();
  }

  //===================================================================================================================//

  template <typename RunnerT>
  int TrainingController<RunnerT>::startTraining()
  {
    if (!this->runner)
      return 1;

    return this->runner->train();
  }

  //===================================================================================================================//
  //-- Accessors --//
  //===================================================================================================================//

  template <typename RunnerT>
  TerminalUI_TrainingWindow* TrainingController<RunnerT>::getWindow() const
  {
    return this->window.get();
  }

  //===================================================================================================================//

  template <typename RunnerT>
  RunnerT* TrainingController<RunnerT>::getRunner() const
  {
    return this->runner.get();
  }

  //===================================================================================================================//
  //-- IRunnerObserver overrides --//
  //===================================================================================================================//

  template <typename RunnerT>
  void TrainingController<RunnerT>::onSampleLoadProgress(ulong current, ulong total, ulong batchIndex,
                                                         ulong totalBatches, bool isValidation)
  {
    (void)batchIndex;
    (void)totalBatches;
    (void)isValidation;

    if (!this->window)
      return;

    QMutexLocker<QRecursiveMutex> lock(&this->window->getMutex());

    float fraction = (total > 0) ? static_cast<float>(current) / static_cast<float>(total) : 0.0f;

    std::string label =
      "Samples " + SummaryTable::formatWithCommas(current) + "/" + SummaryTable::formatWithCommas(total);

    this->window->setLoadingProgress(label, fraction);
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TrainingController<RunnerT>::onValidationProgress(ulong current, ulong total)
  {
    if (!this->window)
      return;

    QMutexLocker<QRecursiveMutex> lock(&this->window->getMutex());

    this->isValidating = true;

    float fraction = (total > 0) ? static_cast<float>(current) / static_cast<float>(total) : 0.0f;
    this->window->updateProgress("Validating", fraction);
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TrainingController<RunnerT>::onBatchProgress(int batchIdx, int totalBatches, float currentLoss,
                                                    float samplesPerSec, float etaSeconds,
                                                    const std::vector<float>& fractions)
  {
    (void)batchIdx;
    (void)totalBatches;

    if (!this->window)
      return;

    QMutexLocker<QRecursiveMutex> lock(&this->window->getMutex());

    // Clear any transitional "Validating" state from the previous epoch.
    this->isValidating = false;

    this->window->updateProgress(this->buildEpochLabel(), fractions);

    // Sub-line: running average loss, ingestion rate, and epoch ETA.
    std::ostringstream stats;
    stats << "Loss: " << std::fixed << std::setprecision(6) << currentLoss << "  " << std::setw(6)
          << static_cast<long>(samplesPerSec) << " img/s  ETA " << formatEta(etaSeconds);
    this->window->updateProgressSubLine(stats.str());

    this->refreshTimingPanel();
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TrainingController<RunnerT>::onEpochCompleted(int epochIdx, int totalEpochs, float epochLoss, bool hasValLoss,
                                                     float valLoss, const std::string& summary)
  {
    if (!this->window)
      return;

    QMutexLocker<QRecursiveMutex> lock(&this->window->getMutex());

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

    bool isBest = summary.find("Best*") != std::string::npos;
    std::string bestStr = isBest ? "✓" : "";
    std::string timestamp = Common::Utils::formatHumanReadable();

    TerminalUI_Table::Row row = {epochStr, lossStream.str(), valLossStr, bestStr, timestamp};

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
  void TrainingController<RunnerT>::onTrainingFinished(bool success, const std::string& finalSummary)
  {
    if (!this->window)
      return;

    QMutexLocker<QRecursiveMutex> lock(&this->window->getMutex());

    this->isValidating = false;

    std::string prefix = success ? "[Training complete] " : "[Training failed] ";
    this->window->addEpochMessage(prefix + finalSummary);
    this->window->refreshEpochContent();
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TrainingController<RunnerT>::onModelInfoUpdated(const std::string& property, const std::string& value)
  {
    (void)property;
    (void)value;

    if (!this->window || !this->runner)
      return;

    QMutexLocker<QRecursiveMutex> lock(&this->window->getMutex());

    // The Runner emits these notifications after it has updated its internal
    // state (e.g. sample counts once the dataset is loaded), so rebuild the
    // whole configuration section from the authoritative row set rather than
    // appending one raw key/value at a time.
    this->window->setModelInfoRows(this->runner->buildModelInfoRows());
    this->window->refreshModelInfoContent();
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TrainingController<RunnerT>::onLogMessage(const std::string& message, bool isError)
  {
    if (!this->window)
      return;

    QMutexLocker<QRecursiveMutex> lock(&this->window->getMutex());

    std::string formatted = isError ? ("[ERROR] " + message) : message;
    this->window->addEpochMessage(formatted);
    this->window->refreshEpochContent();
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TrainingController<RunnerT>::onTimingUpdated(const std::string& metric, float value)
  {
    (void)metric;
    (void)value;

    if (!this->window)
      return;

    QMutexLocker<QRecursiveMutex> lock(&this->window->getMutex());

    this->refreshTimingPanel();
  }

  //===================================================================================================================//
  //-- Private — epoch label --//
  //===================================================================================================================//

  template <typename RunnerT>
  std::string TrainingController<RunnerT>::buildEpochLabel() const
  {
    std::ostringstream oss;
    oss << "Epoch " << std::setw(4) << (this->currentEpoch + 1) << "/" << this->totalEpochs;

    return oss.str();
  }

  //===================================================================================================================//
  //-- Private — timing panel refresh --//
  //===================================================================================================================//

  template <typename RunnerT>
  void TrainingController<RunnerT>::refreshTimingPanel()
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
  //-- Private — model info population --//
  //===================================================================================================================//

  template <typename RunnerT>
  void TrainingController<RunnerT>::populateModelInfo()
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

  template class TrainingController<ANNRunner>;
  template class TrainingController<CNNRunner>;

} // namespace NN_CLI
