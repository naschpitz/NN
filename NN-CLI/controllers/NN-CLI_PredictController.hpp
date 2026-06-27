#ifndef NN_CLI_PREDICTCONTROLLER_HPP
#define NN_CLI_PREDICTCONTROLLER_HPP

#include "NN-CLI_RunnerObserver.hpp"
#include "NN-CLI_TerminalUI_PredictWindow.hpp"

#include <QObject>

#include <atomic>
#include <memory>
#include <string>

class QTimer;

template <typename T>
class QFutureWatcher;

//===================================================================================================================//

namespace NN_CLI
{

  //===================================================================================================================//

  // MVC Controller for prediction sessions.  Bridges a concrete Runner (Model)
  // to TerminalUI_PredictWindow (View) through the IRunnerObserver interface.
  // Owns both components and translates prediction events into view updates.
  //
  // Threading: observer callbacks arrive from worker threads.  Each callback
  // updates view data under the window mutex and returns; all rendering,
  // input polling, and resize handling happen on the main thread via a
  // QTimer driven by the QCoreApplication event loop.  Prediction itself
  // runs on a QtConcurrent worker thread so the main thread is free to
  // service the event loop.
  // Worker threads therefore never touch ncurses and can never be stalled
  // by the terminal.
  //
  // Usage:
  //   auto runner = std::make_unique<ANNRunner>(...);
  //   PredictController<ANNRunner> ctrl;
  //   ctrl.init(std::move(runner));
  //   int result = ctrl.startPredict();

  template <typename RunnerT>
  class PredictController : public IRunnerObserver
  {
    public:
      //-- Ctors / Dtors --//

      PredictController() = default;
      ~PredictController() override;

      PredictController(const PredictController&) = delete;
      PredictController& operator=(const PredictController&) = delete;
      PredictController(PredictController&&) = delete;
      PredictController& operator=(PredictController&&) = delete;

      //-- Lifecycle --//

      // Create the PredictWindow, take ownership of the Runner, and register
      // this controller as an IRunnerObserver on the Runner.
      void init(std::unique_ptr<RunnerT> runner);

      // Trigger the Runner's prediction process.  Returns the exit code from
      // RunnerT::predict().  With the TUI active, prediction runs on a
      // QtConcurrent worker thread while the main thread runs a
      // QCoreApplication event loop (driving the UI timer) until the user
      // dismisses the window.  Without a TUI, prediction runs synchronously.
      int startPredict();

      // Clear observer and destroy window.
      void shutdown();

      //-- Accessors --//

      RunnerT* getRunner() const;
      TerminalUI_PredictWindow* getWindow() const;

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

      void onPredictFinished(const Common::PredictResults<float>& results, size_t numInputs, double durationSeconds,
                             const std::string& durationFormatted, const std::string& outputPath) override;

      void onModelInfoUpdated(const std::string& property, const std::string& value) override;

      void onLogMessage(const std::string& message, bool isError) override;

      void onTimingUpdated(const std::string& metric, float value) override;

    private:
      //-- Methods --//

      // Populate the model info panel with core configuration data.
      void populateModelInfo();

      // Populate the epoch history panel with training metadata.
      void populateTrainMeta();

      // Seed the progress bar with initial state.
      void populateProgress();

      // Check if the user requested abort via the window and forward it to the runner.
      void checkAbortRequested();

      //-- Members --//

      std::unique_ptr<TerminalUI_PredictWindow> window;
      std::unique_ptr<RunnerT> runner;

      //-- Async prediction (TUI path only) --//

      std::unique_ptr<QFutureWatcher<int>> workWatcher;
      std::unique_ptr<QTimer> completionTimer;
      std::atomic<bool> workComplete{false};
      int workResult = 0;

      bool abortHandled = false;

      //-- Qt signal-connection context (thread affinity for Phase 2 queued delivery) --//
      QObject signalContext;
  };

  //===================================================================================================================//

} // namespace NN_CLI

//===================================================================================================================//

#endif // NN_CLI_PREDICTCONTROLLER_HPP
