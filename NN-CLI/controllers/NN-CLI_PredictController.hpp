#ifndef NN_CLI_PREDICTCONTROLLER_HPP
#define NN_CLI_PREDICTCONTROLLER_HPP

#include "NN-CLI_RunnerBase.hpp"
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

  // MVC Controller for prediction sessions.  Bridges a Runner (Model) to
  // TerminalUI_PredictWindow (View) via Qt signals/slots.  Owns both
  // components and translates prediction events into view updates.
  //
  // Threading: prediction runs on a QtConcurrent worker thread while the main
  // thread spins a QCoreApplication event loop (driving the UI timer).  Runner
  // signals are emitted from Core worker threads and queued for delivery on
  // the main thread — no mutex is needed for view data.  Worker threads
  // therefore never touch ncurses and can never be stalled by the terminal.
  //
  // Usage:
  //   auto runner = std::make_unique<ANNRunner>(...);
  //   PredictController ctrl;
  //   ctrl.init(std::move(runner));
  //   int result = ctrl.startPredict();

  class PredictController : public QObject
  {
      Q_OBJECT

    public:
      //-- Ctors / Dtors --//

      PredictController() = default;
      ~PredictController();

      PredictController(const PredictController&) = delete;
      PredictController& operator=(const PredictController&) = delete;
      PredictController(PredictController&&) = delete;
      PredictController& operator=(PredictController&&) = delete;

      //-- Lifecycle --//

      // Create the PredictWindow, take ownership of the Runner, and connect
      // this controller to the Runner's signals.
      void init(std::unique_ptr<RunnerBase> runner);

      // Trigger the Runner's prediction process.  Returns the exit code from
      // Runner::predict().  With the TUI active, prediction runs on a
      // QtConcurrent worker thread while the main thread runs a
      // QCoreApplication event loop (driving the UI timer) until the user
      // dismisses the window.  Without a TUI, prediction runs synchronously.
      int startPredict();

      // Disconnect signals and destroy window.
      void shutdown();

      //-- Accessors --//

      RunnerBase* getRunner() const;
      TerminalUI_PredictWindow* getWindow() const;

    public slots:
      //-- Runner signal handlers --//

      void onSampleLoadProgress(ulong current, ulong total, ulong batchIndex, ulong totalBatches, bool isValidation);

      void onValidationProgress(ulong current, ulong total);

      void onBatchProgress(int batchIdx, int totalBatches, float currentLoss, float samplesPerSec, float etaSeconds,
                           const std::vector<float>& fractions);

      void onEpochCompleted(int epochIdx, int totalEpochs, float epochLoss, bool hasValLoss, float valLoss,
                            float learningRate, const std::string& summary);

      void onTrainFinished(bool success, const std::string& finalSummary);

      void onPredictFinished(const Common::PredictResults<float>& results, size_t numInputs, double durationSeconds,
                             const std::string& durationFormatted, const std::string& outputPath);

      void onModelInfoUpdated(const std::string& property, const std::string& value);

      void onLogMessage(const std::string& message, bool isError);

      void onTimingUpdated(const std::string& metric, float value);

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
      std::unique_ptr<RunnerBase> runner;

      //-- Async prediction (TUI path only) --//

      std::unique_ptr<QFutureWatcher<int>> workWatcher;
      std::unique_ptr<QTimer> completionTimer;
      std::atomic<bool> workComplete{false};
      int workResult = 0;

      bool abortHandled = false;
  };

  //===================================================================================================================//

} // namespace NN_CLI

//===================================================================================================================//

#endif // NN_CLI_PREDICTCONTROLLER_HPP
