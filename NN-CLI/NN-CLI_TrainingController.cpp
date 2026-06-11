#include "NN-CLI_TrainingController.hpp"

#include "NN-CLI_ANNRunner.hpp"
#include "NN-CLI_CNNRunner.hpp"
#include "NN-CLI_TerminalUI_TrainingWindow.hpp"

#include <ANN_Utils.hpp>

#include "Common/Common_CostFunctionConfig.hpp"
#include "Common/Common_Optimizer.hpp"

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
  //-- Loading progress --//
  //===================================================================================================================//

  template <typename RunnerT>
  void TrainingController<RunnerT>::startLoadingProgress(const std::string& label)
  {
    if (!this->window)
      return;

    this->window->updateProgress(label, 0.0f);
    this->window->draw();
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TrainingController<RunnerT>::stopLoadingProgress()
  {
    if (!this->window)
      return;

    this->window->clearLoadingProgress();
    this->window->draw();
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
  void TrainingController<RunnerT>::onBatchProgress(int batchIdx, int totalBatches, float currentLoss,
                                                      const std::vector<float>& fractions)
  {
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
    std::string bestStr = isBest ? "Best*" : "";
    std::string timestamp = ANN::Utils<float>::formatISO8601();

    TerminalUI_Table::Row row = {epochStr, lossStream.str(), accuracyStr, bestStr, timestamp};

    this->window->addEpochRow(row);
    this->window->addEpochMessage(summary);
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
    if (!this->window)
      return;

    this->window->addModelInfoEntry(property, value);
    this->window->refreshModelInfoContent();
    this->window->draw();
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TrainingController<RunnerT>::onLogMessage(const std::string& message, bool isError)
  {
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

    const auto& coreConfig = this->runner->getCoreConfig();

    //-- Optimizer --//
    std::string optimizerName = Common::Optimizer<float>::typeToName(coreConfig.trainingConfig.optimizer.type);
    this->window->addModelInfoEntry("Optimizer", optimizerName);

    //-- Learning rate --//
    std::ostringstream lrStream;
    lrStream << std::fixed << std::setprecision(6) << coreConfig.trainingConfig.learningRate;
    this->window->addModelInfoEntry("Learning Rate", lrStream.str());

    //-- Batch size --//
    this->window->addModelInfoEntry("Batch Size", std::to_string(coreConfig.trainingConfig.batchSize));

    //-- Dropout rate --//
    std::ostringstream dropoutStream;
    dropoutStream << std::fixed << std::setprecision(2) << coreConfig.trainingConfig.dropoutRate;
    this->window->addModelInfoEntry("Dropout Rate", dropoutStream.str());

    //-- Cost function --//
    std::string costFunctionName = Common::CostFunction::typeToName(coreConfig.costFunctionConfig.type);
    this->window->addModelInfoEntry("Cost Function", costFunctionName);

    this->window->refreshModelInfoContent();
  }

  //===================================================================================================================//
  //-- Explicit template instantiations --//
  //===================================================================================================================//

  template class TrainingController<ANNRunner>;
  template class TrainingController<CNNRunner>;

} // namespace NN_CLI
