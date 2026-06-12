#ifndef NN_CLI_PREDICTCONTROLLER_HPP
#define NN_CLI_PREDICTCONTROLLER_HPP

#include "NN-CLI_RunnerObserver.hpp"

#include <memory>
#include <string>

//===================================================================================================================//

namespace NN_CLI
{

  //===================================================================================================================//

  // MVC Controller for prediction sessions.  Bridges a concrete Runner (Model)
  // to console output (View) through the IRunnerObserver interface.  Takes
  // ownership of the runner, registers itself as an observer, and delegates
  // runner events to stdout/stderr.
  //
  // Template parameter RunnerT is the concrete runner type (e.g. ANNRunner or
  // CNNRunner).  The controller calls RunnerT::predict() and prints progress
  // and results to the console.
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

      // Take ownership of the Runner and register this controller as an
      // IRunnerObserver on the Runner.
      void init(std::unique_ptr<RunnerT> runner);

      // Trigger the Runner's prediction process.  Returns the exit code from
      // RunnerT::predict().
      int startPredict();

      //-- Accessors --//

      RunnerT* getRunner() const;

    protected:
      //-- IRunnerObserver overrides --//

      void onBatchProgress(int batchIdx, int totalBatches, float currentLoss, float samplesPerSec,
                           float etaSeconds, const std::vector<float>& fractions) override;

      void onEpochCompleted(int epochIdx, int totalEpochs, float epochLoss, bool hasValLoss, float valLoss,
                            const std::string& summary) override;

      void onTrainingFinished(bool success, const std::string& finalSummary) override;

      void onModelInfoUpdated(const std::string& property, const std::string& value) override;

      void onLogMessage(const std::string& message, bool isError) override;

      void onTimingUpdated(const std::string& metric, float value) override;

    private:
      //-- Members --//

      std::unique_ptr<RunnerT> runner;
  };

  //===================================================================================================================//

} // namespace NN_CLI

//===================================================================================================================//

#endif // NN_CLI_PREDICTCONTROLLER_HPP
