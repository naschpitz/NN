#ifndef NN_CLI_TESTCONTROLLER_HPP
#define NN_CLI_TESTCONTROLLER_HPP

#include "NN-CLI_RunnerBase.hpp"

#include <QObject>

#include <memory>
#include <string>

//===================================================================================================================//

namespace NN_CLI
{

  //===================================================================================================================//

  // MVC Controller for test/evaluation sessions.  Bridges a Runner (Model) to
  // console output (View) via Qt signals/slots.  Takes ownership of the
  // runner, connects its signals, and delegates runner events to stdout/stderr.
  //
  // The controller calls Runner::test() and prints test metrics and results
  // to the console.  Uses Qt::DirectConnection because the main thread blocks
  // synchronously in test() — there is no event loop to queue into.
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

      // Take ownership of the Runner and connect this controller to the
      // Runner's signals.
      void init(std::unique_ptr<RunnerBase> runner);

      // Trigger the Runner's test process.  Returns the exit code from
      // Runner::test().
      int startTest();

      //-- Accessors --//

      RunnerBase* getRunner() const;

    public slots:
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

      std::unique_ptr<RunnerBase> runner;
  };

  //===================================================================================================================//

} // namespace NN_CLI

//===================================================================================================================//

#endif // NN_CLI_TESTCONTROLLER_HPP
