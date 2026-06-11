#include "NN-CLI_TrainingController.hpp"

#include "NN-CLI_ANNRunner.hpp"
#include "NN-CLI_CNNRunner.hpp"
#include "NN-CLI_TerminalUI_TrainingWindow.hpp"

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

    if (this->runner)
      this->runner->addObserver(this);

    // Initialize the ncurses TUI.  If init fails (e.g. no TTY attached),
    // the window gracefully degrades -- draw() becomes a no-op, so the
    // training proceeds with console-only output from the Runner.
    if (this->window)
      this->window->init();
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
  void TrainingController<RunnerT>::onBatchProgress(int batchIdx, int totalBatches, float currentLoss, float fraction)
  {
    if (!this->window)
      return;

    std::string label = "Batch " + std::to_string(batchIdx) + "/" + std::to_string(totalBatches) +
                        "  Loss: " + std::to_string(currentLoss);
    this->window->updateProgress(label, fraction);
    this->window->draw();
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TrainingController<RunnerT>::onEpochCompleted(int epochIdx, int totalEpochs, float epochLoss, float accuracy,
                                                      const std::string& summary)
  {
    if (!this->window)
      return;

    this->window->addEpochMessage(summary);
    this->window->refreshEpochContent();
    this->window->draw();
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TrainingController<RunnerT>::onTrainingFinished(bool success, const std::string& finalSummary)
  {
    if (!this->window)
      return;

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
    if (!this->window)
      return;

    this->timingMetrics[metric] = value;

    std::vector<std::string> lines;
    lines.reserve(this->timingMetrics.size());

    for (const auto& [name, val] : this->timingMetrics)
      lines.push_back(" " + name + ": " + std::to_string(val) + " ms");

    this->window->setTimingLines(lines);
    this->window->refreshTimingContent();
    this->window->draw();
  }

  //===================================================================================================================//
  //-- Explicit template instantiations --//
  //===================================================================================================================//

  template class TrainingController<ANNRunner>;
  template class TrainingController<CNNRunner>;

} // namespace NN_CLI
