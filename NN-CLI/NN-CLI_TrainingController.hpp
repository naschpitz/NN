#ifndef NN_CLI_TRAININGCONTROLLER_HPP
#define NN_CLI_TRAININGCONTROLLER_HPP

#include "NN-CLI_RunnerObserver.hpp"
#include "NN-CLI_TerminalUI_TrainingWindow.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace NN_CLI
{

  //===================================================================================================================//

  // MVC Controller for training sessions.  Bridges a concrete Runner (Model)
  // and a TerminalUI_TrainingWindow (View) through the IRunnerObserver
  // interface.  Owns both components and translates training events into
  // high-level view updates — the controller itself is completely free of
  // ncurses internals.
  //
  // Template parameter RunnerT is the concrete runner type (e.g. ANNRunner or
  // CNNRunner).  The controller takes ownership of the runner via unique_ptr
  // and registers itself as an observer to receive batch, epoch, and model-info
  // events.  Each observer override delegates to a single high-level call on
  // the TrainingWindow, keeping the mapping transparent and testable.
  //
  // Usage:
  //   auto runner = std::make_unique<ANNRunner>(...);
  //   TrainingController<ANNRunner> ctrl;
  //   ctrl.init(std::move(runner));
  //   int result = ctrl.startTraining();

  template <typename RunnerT>
  class TrainingController : public IRunnerObserver
  {
    public:
      //-- Ctors / Dtors --//

      TrainingController() = default;

      ~TrainingController() override;

      TrainingController(const TrainingController&) = delete;
      TrainingController& operator=(const TrainingController&) = delete;
      TrainingController(TrainingController&&) = delete;
      TrainingController& operator=(TrainingController&&) = delete;

      //-- Lifecycle --//

      // Create the TrainingWindow, take ownership of the Runner, and register
      // this controller as an IRunnerObserver on the Runner.
      void init(std::unique_ptr<RunnerT> runner);

      // Trigger the Runner's training process.  Returns the exit code from
      // RunnerT::train().
      int startTraining();

      //-- Accessors --//

      TerminalUI_TrainingWindow* getWindow() const;
      RunnerT* getRunner() const;

    protected:
      //-- IRunnerObserver overrides --//

      void onSampleLoadProgress(ulong current, ulong total, ulong batchIndex, ulong totalBatches,
                                bool isValidation) override;

      void onValidationProgress(ulong current, ulong total) override;

      void onBatchProgress(int batchIdx, int totalBatches, float currentLoss,
                           const std::vector<float>& fractions) override;

      void onEpochCompleted(int epochIdx, int totalEpochs, float epochLoss, float accuracy,
                            const std::string& summary) override;

      void onTrainingFinished(bool success, const std::string& finalSummary) override;

      void onModelInfoUpdated(const std::string& property, const std::string& value) override;

      void onLogMessage(const std::string& message, bool isError) override;

      void onTimingUpdated(const std::string& metric, float value) override;

    private:
      //-- Methods --//

      // Populate the model info panel with core configuration data.
      void populateModelInfo();

      // Refresh the timing panel content from the runner's profiling data.
      void refreshTimingPanel();

      //-- Members --//

      std::unique_ptr<TerminalUI_TrainingWindow> window;
      std::unique_ptr<RunnerT> runner;

      //-- Training state --//

      int currentEpoch = 0;
      int totalEpochs = 0;
      bool isValidating = false;

      // Serializes all View (ncurses) access.  Observer callbacks arrive from
      // multiple worker threads — the per-batch training callback, the data
      // loader's loading callback, and the validation callback — so every
      // window mutation + draw must hold this lock.  Recursive so a handler
      // that fans out to helpers cannot self-deadlock.
      std::recursive_mutex uiMutex;
  };

  //===================================================================================================================//

} // namespace NN_CLI

#endif // NN_CLI_TRAININGCONTROLLER_HPP
