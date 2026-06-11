#include "NN-CLI_TrainingController.hpp"

#include "NN-CLI_ANNRunner.hpp"
#include "NN-CLI_CNNRunner.hpp"
#include "NN-CLI_TerminalUI_TrainingWindow.hpp"
#include "NN-CLI_LossReferenceTable.hpp"

#include "Common/Common_Utils.hpp"

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

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
    // the window gracefully degrades -- draw() becomes a no-op, so the
    // training proceeds with console-only output from the Runner.
    if (this->window)
      this->window->init();

    // Populate the Model Info panel with static core configuration data.
    this->populateModelInfo();
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
    (void)isValidation;

    std::lock_guard<std::recursive_mutex> lock(this->uiMutex);

    if (!this->window)
      return;

    float fraction = (total > 0) ? static_cast<float>(current) / static_cast<float>(total) : 0.0f;

    std::string label = "Samples";
    if (totalBatches > 0)
      label += " (" + std::to_string(batchIndex + 1) + "/" + std::to_string(totalBatches) + ")";

    this->window->setLoadingProgress(label, fraction);
    this->window->draw();
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TrainingController<RunnerT>::onValidationProgress(ulong current, ulong total)
  {
    std::lock_guard<std::recursive_mutex> lock(this->uiMutex);

    if (!this->window)
      return;

    this->isValidating = true;

    float fraction = (total > 0) ? static_cast<float>(current) / static_cast<float>(total) : 0.0f;
    this->window->updateProgress("Validating", fraction);
    this->window->draw();
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TrainingController<RunnerT>::onBatchProgress(int batchIdx, int totalBatches, float currentLoss,
                                                       const std::vector<float>& fractions)
  {
    std::lock_guard<std::recursive_mutex> lock(this->uiMutex);

    if (!this->window)
      return;

    // Clear any transitional "Validating" state from the previous epoch.
    this->isValidating = false;

    int displayEpoch = this->currentEpoch + 1;
    std::string label = "Training (" + std::to_string(displayEpoch) + "/" + std::to_string(this->totalEpochs) + ")";
    this->window->updateProgress(label, fractions);
    this->refreshTimingPanel();
    this->window->draw();
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TrainingController<RunnerT>::onEpochCompleted(int epochIdx, int totalEpochs, float epochLoss, float accuracy,
                                                       const std::string& summary)
  {
    std::lock_guard<std::recursive_mutex> lock(this->uiMutex);

    if (!this->window)
      return;

    // Track the current epoch (0-based index → next epoch number for display).
    this->currentEpoch = epochIdx + 1;

    // Build a structured table row for the epoch data.
    std::string epochStr = std::to_string(epochIdx + 1);

    std::ostringstream lossStream;
    lossStream << std::fixed << std::setprecision(6) << epochLoss;

    std::string accuracyStr;
    if (accuracy >= 0.0f) {
      std::ostringstream accStream;
      accStream << std::fixed << std::setprecision(2) << accuracy;
      accuracyStr = accStream.str();
    } else {
      accuracyStr = "N/A";
    }

    bool isBest = summary.find("Best*") != std::string::npos;
    std::string bestStr = isBest ? "✓" : "";
    std::string timestamp = Common::Utils::formatHumanReadable();

    TerminalUI_Table::Row row = {epochStr, lossStream.str(), accuracyStr, bestStr, timestamp};

    this->window->addEpochRow(row);
    this->window->refreshEpochContent();

    // When validation was performed for this epoch (accuracy >= 0), show a
    // transitional "Validating" progress bar that persists until the next
    // onBatchProgress event replaces it with "Training".  The "Validating"
    // bar replaces the progress bar content, NOT the panel title.
    if (accuracy >= 0.0f) {
      this->isValidating = true;
      this->window->updateProgress("Validating", 1.0f);
    }

    this->window->draw();
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TrainingController<RunnerT>::onTrainingFinished(bool success, const std::string& finalSummary)
  {
    std::lock_guard<std::recursive_mutex> lock(this->uiMutex);

    if (!this->window)
      return;

    this->isValidating = false;

    std::string prefix = success ? "[Training complete] " : "[Training failed] ";
    this->window->addEpochMessage(prefix + finalSummary);
    this->window->refreshEpochContent();
    this->window->draw();
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TrainingController<RunnerT>::onModelInfoUpdated(const std::string& property, const std::string& value)
  {
    (void)property;
    (void)value;

    std::lock_guard<std::recursive_mutex> lock(this->uiMutex);

    if (!this->window || !this->runner)
      return;

    // The Runner emits these notifications after it has updated its internal
    // state (e.g. sample counts once the dataset is loaded), so rebuild the
    // whole configuration section from the authoritative row set rather than
    // appending one raw key/value at a time.
    this->window->setModelInfoRows(this->runner->buildModelInfoRows());
    this->window->refreshModelInfoContent();
    this->window->draw();
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TrainingController<RunnerT>::onLogMessage(const std::string& message, bool isError)
  {
    std::lock_guard<std::recursive_mutex> lock(this->uiMutex);

    if (!this->window)
      return;

    std::string formatted = isError ? ("[ERROR] " + message) : message;
    this->window->addEpochMessage(formatted);
    this->window->refreshEpochContent();
    this->window->draw();
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TrainingController<RunnerT>::onTimingUpdated(const std::string& metric, float value)
  {
    (void)metric;
    (void)value;

    std::lock_guard<std::recursive_mutex> lock(this->uiMutex);

    if (!this->window)
      return;

    this->refreshTimingPanel();
    this->window->draw();
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
