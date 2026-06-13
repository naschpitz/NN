#ifndef NN_CLI_CALIBRATECONTROLLER_HPP
#define NN_CLI_CALIBRATECONTROLLER_HPP

#include "NN-CLI_CalibrateUtils.hpp"
#include "NN-CLI_RunnerObserver.hpp"

#include <memory>
#include <string>

//===================================================================================================================//

namespace NN_CLI
{

  //===================================================================================================================//

  /**
   * MVC Controller for calibration sessions.  Bridges a concrete Runner (Model)
   * and console output (View) through the IRunnerObserver interface.  Takes
   * ownership of the runner, registers itself as an observer, and delegates
   * runner events to stdout/stderr.
   *
   * Template parameter RunnerT is the concrete runner type (e.g. ANNRunner or
    * CNNRunner).  The controller calls RunnerT::calibrate() and prints
   * progress and results to the console.
   *
   * Usage:
   *   auto runner = std::make_unique<ANNRunner>(...);
   *   CalibrateController<ANNRunner> ctrl;
   *   ctrl.init(std::move(runner));
    *   int result = ctrl.startCalibrate();
   */
  template <typename RunnerT>
  class CalibrateController : public IRunnerObserver
  {
    public:
      //-- Ctors / Dtors --//

      CalibrateController() = default;
      ~CalibrateController() override;

      CalibrateController(const CalibrateController&) = delete;
      CalibrateController& operator=(const CalibrateController&) = delete;
      CalibrateController(CalibrateController&&) = delete;
      CalibrateController& operator=(CalibrateController&&) = delete;

      //-- Lifecycle --//

      // Take ownership of the Runner and register this controller as an
      // IRunnerObserver on the Runner.
      void init(std::unique_ptr<RunnerT> runner);

      // Trigger the Runner's calibration process.  Returns the exit code from
      // RunnerT::calibrate().
      int startCalibrate();

      //-- Accessors --//

      RunnerT* getRunner() const;

    protected:
      //-- IRunnerObserver overrides --//

      void onBatchProgress(int batchIdx, int totalBatches, float currentLoss, float samplesPerSec, float etaSeconds,
                           const std::vector<float>& fractions) override;

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

#endif // NN_CLI_CALIBRATECONTROLLER_HPP
