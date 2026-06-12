#ifndef NN_CLI_TRAININGCONTROLLER_HPP
#define NN_CLI_TRAININGCONTROLLER_HPP

#include "NN-CLI_RunnerObserver.hpp"
#include "NN-CLI_TerminalUI_TrainingWindow.hpp"

#include <memory>
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
  // Threading: observer callbacks arrive from multiple worker threads (the
  // per-batch training callback, the data loader's loading callback, the
  // validation callback).  Each callback only updates view data under the
  // window's mutex and returns; all rendering, input polling, and resize
  // handling happen on the window's dedicated UI thread (started in init()).
  // Worker threads therefore never touch ncurses and can never be stalled
  // by the terminal.
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

      // Build the training progress bar label for the epoch currently in
      // progress (e.g. "Epoch    1/100").
      std::string buildEpochLabel() const;

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
  };

  //===================================================================================================================//

} // namespace NN_CLI

#endif // NN_CLI_TRAININGCONTROLLER_HPP
