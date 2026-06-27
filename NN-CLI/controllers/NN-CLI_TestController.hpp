#ifndef NN_CLI_TESTCONTROLLER_HPP
#define NN_CLI_TESTCONTROLLER_HPP

#include "NN-CLI_RunnerSignals.hpp"

#include <QObject>

#include <memory>
#include <string>

//===================================================================================================================//

namespace NN_CLI
{

  //===================================================================================================================//

  // MVC Controller for test/evaluation sessions.  Bridges a concrete Runner
  // (Model) to console output (View) via Runner signals.  Takes ownership of
  // the runner, connects its signals, and delegates runner events to
  // stdout/stderr.
  //
  // Template parameter RunnerT is the concrete runner type (e.g. ANNRunner or
  // CNNRunner).  The controller calls RunnerT::test() and prints test metrics
  // and results to the console.
  //
  // Usage:
  //   auto runner = std::make_unique<ANNRunner>(...);
  //   TestController<ANNRunner> ctrl;
  //   ctrl.init(std::move(runner));
  //   int result = ctrl.startTest();

  template <typename RunnerT>
  class TestController
  {
    public:
      //-- Ctors / Dtors --//

      TestController() = default;
      ~TestController();

      TestController(const TestController&) = delete;
      TestController& operator=(const TestController&) = delete;
      TestController(TestController&&) = delete;
      TestController& operator=(TestController&&) = delete;

      //-- Lifecycle --//

      // Take ownership of the Runner and connect this controller to the
      // Runner's signals.
      void init(std::unique_ptr<RunnerT> runner);

      // Trigger the Runner's test process.  Returns the exit code from
      // RunnerT::test().
      int startTest();

      //-- Accessors --//

      RunnerT* getRunner() const;

    protected:
      //-- Runner signal handlers --//

      void onBatchProgress(int batchIdx, int totalBatches, float currentLoss, float samplesPerSec, float etaSeconds,
                           const std::vector<float>& fractions);

      void onEpochCompleted(int epochIdx, int totalEpochs, float epochLoss, bool hasValLoss, float valLoss,
                            float learningRate, const std::string& summary);

      void onTrainFinished(bool success, const std::string& finalSummary);

      void onModelInfoUpdated(const std::string& property, const std::string& value);

      void onLogMessage(const std::string& message, bool isError);

      void onTimingUpdated(const std::string& metric, float value);

    private:
      //-- Members --//

      std::unique_ptr<RunnerT> runner;

      //-- Qt signal-connection context (thread affinity for queued delivery) --//
      QObject signalContext;
  };

  //===================================================================================================================//

} // namespace NN_CLI

//===================================================================================================================//

#endif // NN_CLI_TESTCONTROLLER_HPP
