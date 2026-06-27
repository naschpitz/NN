#ifndef NN_CLI_CALIBRATECONTROLLER_HPP
#define NN_CLI_CALIBRATECONTROLLER_HPP

#include "NN-CLI_CalibrateUtils.hpp"
#include "NN-CLI_RunnerSignals.hpp"

#include <QObject>

#include <memory>
#include <string>

//===================================================================================================================//

namespace NN_CLI
{

  //===================================================================================================================//

  /**
   * MVC Controller for calibration sessions.  Bridges a concrete Runner (Model)
   * and console output (View) via Runner signals.  Takes ownership of the
   * runner, connects its signals, and delegates runner events to stdout/stderr.
   *
   * Template parameter RunnerT is the concrete runner type (e.g. ANNRunner or
   * CNNRunner).  The controller calls RunnerT::calibrate() and prints progress
   * and results to the console.
   *
   * Usage:
   *   auto runner = std::make_unique<ANNRunner>(...);
   *   CalibrateController<ANNRunner> ctrl;
   *   ctrl.init(std::move(runner));
   *   int result = ctrl.startCalibrate();
   */
  template <typename RunnerT>
  class CalibrateController
  {
    public:
      //-- Ctors / Dtors --//

      CalibrateController() = default;
      ~CalibrateController();

      CalibrateController(const CalibrateController&) = delete;
      CalibrateController& operator=(const CalibrateController&) = delete;
      CalibrateController(CalibrateController&&) = delete;
      CalibrateController& operator=(CalibrateController&&) = delete;

      //-- Lifecycle --//

      // Take ownership of the Runner and connect this controller to the
      // Runner's signals.
      void init(std::unique_ptr<RunnerT> runner);

      // Trigger the Runner's calibration process.  Returns the exit code from
      // RunnerT::calibrate().
      int startCalibrate();

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

#endif // NN_CLI_CALIBRATECONTROLLER_HPP
