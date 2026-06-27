#ifndef NN_CLI_TRAINCONTROLLER_HPP
#define NN_CLI_TRAINCONTROLLER_HPP

#include "NN-CLI_RunnerObserver.hpp"
#include "NN-CLI_TerminalUI_TrainWindow.hpp"

#include <QObject>

#include <atomic>
#include <memory>
#include <string>
#include <vector>

class QTimer;

template <typename T>
class QFutureWatcher;

namespace NN_CLI
{

  //===================================================================================================================//

  // MVC Controller for training sessions.  Bridges a concrete Runner (Model)
  // and a TerminalUI_TrainWindow (View) through the IRunnerObserver
  // interface.  Owns both components and translates training events into
  // high-level view updates — the controller itself is completely free of
  // ncurses internals.
  //
  // Threading: observer callbacks arrive from multiple worker threads (the
  // per-batch training callback, the data loader's loading callback, the
  // validation callback).  Each callback only updates view data under the
  // window's mutex and returns; all rendering, input polling, and resize
  // handling happen on the main thread via a QTimer driven by the
  // QCoreApplication event loop.  Training itself runs on a QtConcurrent
  // worker thread so the main thread is free to service the event loop.
  // Worker threads therefore never touch ncurses and can never be stalled
  // by the terminal.
  //
  // Template parameter RunnerT is the concrete runner type (e.g. ANNRunner or
  // CNNRunner).  The controller takes ownership of the runner via unique_ptr
  // and registers itself as an observer to receive batch, epoch, and model-info
  // events.  Each observer override delegates to a single high-level call on
  // the TrainWindow, keeping the mapping transparent and testable.
  //
  // Usage:
  //   auto runner = std::make_unique<ANNRunner>(...);
  //   TrainController<ANNRunner> ctrl;
  //   ctrl.init(std::move(runner));
  //   int result = ctrl.startTrain();

  template <typename RunnerT>
  class TrainController : public IRunnerObserver
  {
    public:
      //-- Ctors / Dtors --//

      TrainController() = default;

      ~TrainController() override;

      TrainController(const TrainController&) = delete;
      TrainController& operator=(const TrainController&) = delete;
      TrainController(TrainController&&) = delete;
      TrainController& operator=(TrainController&&) = delete;

      //-- Lifecycle --//

      // Create the TrainWindow, take ownership of the Runner, and register
      // this controller as an IRunnerObserver on the Runner.
      void init(std::unique_ptr<RunnerT> runner);

      // Trigger the Runner's training process.  Returns the exit code from
      // RunnerT::train().  With the TUI active, training runs on a
      // QtConcurrent worker thread while the main thread runs a
      // QCoreApplication event loop (driving the UI timer) until the user
      // dismisses the window.  Without a TUI, training runs synchronously.
      int startTrain();

      //-- Accessors --//

      TerminalUI_TrainWindow* getWindow() const;
      RunnerT* getRunner() const;

    protected:
      //-- IRunnerObserver overrides --//

      void onSampleLoadProgress(ulong current, ulong total, ulong batchIndex, ulong totalBatches,
                                bool isValidation) override;

      void onValidationProgress(ulong current, ulong total) override;

      void onBatchProgress(int batchIdx, int totalBatches, float currentLoss, float samplesPerSec, float etaSeconds,
                           const std::vector<float>& fractions) override;

      void onEpochCompleted(int epochIdx, int totalEpochs, float epochLoss, bool hasValLoss, float valLoss,
                            float learningRate, const std::string& summary) override;

      void onTrainFinished(bool success, const std::string& finalSummary) override;

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

      // Check the window's abort flag and forward it to the runner.
      void checkAbortRequested();

      //-- Members --//

      std::unique_ptr<TerminalUI_TrainWindow> window;
      std::unique_ptr<RunnerT> runner;

      //-- Async training (TUI path only) --//

      std::unique_ptr<QFutureWatcher<int>> workWatcher;
      std::unique_ptr<QTimer> completionTimer;
      std::atomic<bool> workComplete{false};
      int workResult = 0;

      //-- Training state --//

      int currentEpoch = 0;
      int totalEpochs = 0;
      bool isValidating = false;
      bool abortHandled = false;

      //-- Qt signal-connection context (thread affinity for Phase 2 queued delivery) --//
      QObject signalContext;
  };

  //===================================================================================================================//

} // namespace NN_CLI

#endif // NN_CLI_TRAINCONTROLLER_HPP
