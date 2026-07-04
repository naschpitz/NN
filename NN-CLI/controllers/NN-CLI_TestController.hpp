#ifndef NN_CLI_TESTCONTROLLER_HPP
#define NN_CLI_TESTCONTROLLER_HPP

#include "NN-CLI_RunnerBase.hpp"
#include "NN-CLI_TerminalUI_TestWindow.hpp"

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

  // MVC Controller for test/evaluation sessions.  Bridges a Runner (Model) to
  // TerminalUI_TestWindow (View) via Qt signals/slots.  Owns both components
  // and translates test events into view updates.
  //
  // Threading: test runs on a QtConcurrent worker thread while the main thread
  // spins a QCoreApplication event loop (driving the UI timer).  Runner
  // signals are emitted from Core worker threads and queued for delivery on
  // the main thread — no mutex is needed for view data.  Worker threads
  // therefore never touch ncurses and can never be stalled by the terminal.
  //
  // Without a TUI (e.g. piped stdin in headless test runs) test runs
  // synchronously and the Runner owns the console output.
  //
  // Usage:
  //   auto runner = std::make_unique<ANNRunner>(...);
  //   TestController ctrl;
  //   ctrl.init(std::move(runner));
  //   int result = ctrl.startTest();

  class TestController : public QObject
  {
      Q_OBJECT

    public:
      //-- Ctors / Dtors --//

      TestController() = default;
      ~TestController();

      TestController(const TestController&) = delete;
      TestController& operator=(const TestController&) = delete;
      TestController(TestController&&) = delete;
      TestController& operator=(TestController&&) = delete;

      //-- Lifecycle --//

      // Create the TestWindow, take ownership of the Runner, and connect this
      // controller to the Runner's signals.
      void init(std::unique_ptr<RunnerBase> runner);

      // Trigger the Runner's test process.  Returns the exit code from
      // Runner::test().  With the TUI active, test runs on a QtConcurrent
      // worker thread while the main thread runs a QCoreApplication event
      // loop (driving the UI timer) until the user dismisses the window.
      // Without a TUI, test runs synchronously.
      int startTest();

      // Disconnect signals and destroy window.
      void shutdown();

      //-- Accessors --//

      RunnerBase* getRunner() const;
      TerminalUI_TestWindow* getWindow() const;

    public slots:
      //-- Runner signal handlers --//

      void onSampleLoadProgress(ulong current, ulong total, ulong batchIndex, ulong totalBatches, bool isValidation);

      void onBatchProgress(int batchIdx, int totalBatches, float currentLoss, float samplesPerSec, float etaSeconds,
                           const std::vector<float>& fractions);

      void onTestFinished(const Common::TestResult<float>& result, double durationSeconds,
                          const std::string& durationFormatted, const std::string& outputPath);

      void onModelInfoUpdated(const std::string& property, const std::string& value);

      void onLogMessage(const std::string& message, bool isError);

      void onTimingUpdated(const std::string& metric, float value);

    private:
      //-- Methods --//

      // Populate the model info panel with core configuration data.
      void populateModelInfo();

      // Populate the epoch history panel with the loaded model's training metadata.
      void populateTrainMeta();

      // Seed the progress bar with initial state.
      void populateProgress();

      // Check if the user requested abort via the window and forward it to the runner.
      void checkAbortRequested();

      //-- Members --//

      std::unique_ptr<TerminalUI_TestWindow> window;
      std::unique_ptr<RunnerBase> runner;

      //-- Async test (TUI path only) --//

      std::unique_ptr<QFutureWatcher<int>> workWatcher;
      std::unique_ptr<QTimer> completionTimer;
      std::atomic<bool> workComplete{false};
      int workResult = 0;

      bool abortHandled = false;
  };

  //===================================================================================================================//

} // namespace NN_CLI

//===================================================================================================================//

#endif // NN_CLI_TESTCONTROLLER_HPP
