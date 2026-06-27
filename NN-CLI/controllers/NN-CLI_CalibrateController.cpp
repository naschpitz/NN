#include "NN-CLI_CalibrateController.hpp"

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace NN_CLI
{

  //===================================================================================================================//
  //-- Ctors / Dtors --//
  //===================================================================================================================//

  CalibrateController::~CalibrateController()
  {
    // Signal connections auto-disconnect when the Runner (sender) is destroyed.
  }

  //===================================================================================================================//
  //-- Lifecycle --//
  //===================================================================================================================//

  void CalibrateController::init(std::unique_ptr<RunnerBase> runner)
  {
    this->runner = std::move(runner);

    if (!this->runner)
      return;

    // Calibration runs synchronously with no event loop.  DirectConnection
    // delivers the slots inline on the emitting thread (same threading
    // semantics as the original observer pattern), so no queued events pile up
    // while the main thread is blocked in runner->calibrate().
    QObject::connect(this->runner.get(), &RunnerBase::batchProgress, this, &CalibrateController::onBatchProgress,
                     Qt::DirectConnection);
    QObject::connect(this->runner.get(), &RunnerBase::epochCompleted, this, &CalibrateController::onEpochCompleted,
                     Qt::DirectConnection);
    QObject::connect(this->runner.get(), &RunnerBase::trainFinished, this, &CalibrateController::onTrainFinished,
                     Qt::DirectConnection);
    QObject::connect(this->runner.get(), &RunnerBase::modelInfoUpdated, this, &CalibrateController::onModelInfoUpdated,
                     Qt::DirectConnection);
    QObject::connect(this->runner.get(), &RunnerBase::logMessage, this, &CalibrateController::onLogMessage,
                     Qt::DirectConnection);
    QObject::connect(this->runner.get(), &RunnerBase::timingUpdated, this, &CalibrateController::onTimingUpdated,
                     Qt::DirectConnection);
  }

  //===================================================================================================================//

  int CalibrateController::startCalibrate()
  {
    if (!this->runner)
      return 1;

    return this->runner->calibrate();
  }

  //===================================================================================================================//
  //-- Accessors --//
  //===================================================================================================================//

  RunnerBase* CalibrateController::getRunner() const
  {
    return this->runner.get();
  }

  //===================================================================================================================//
  //-- Runner signal handlers --//
  //===================================================================================================================//

  void CalibrateController::onBatchProgress(int batchIdx, int totalBatches, float currentLoss, float samplesPerSec,
                                            float etaSeconds, const std::vector<float>& fractions)
  {
    // Calibrate mode does not use batch progress events, but print for
    // interface completeness in case the runner fires one.
    (void)batchIdx;
    (void)totalBatches;
    (void)currentLoss;
    (void)samplesPerSec;
    (void)etaSeconds;
    (void)fractions;
  }

  //===================================================================================================================//

  void CalibrateController::onEpochCompleted(int epochIdx, int totalEpochs, float epochLoss, bool hasValLoss,
                                             float valLoss, float learningRate, const std::string& summary)
  {
    // Calibrate mode does not use epoch events, but print the summary for
    // interface completeness in case the runner fires one.
    (void)epochIdx;
    (void)totalEpochs;
    (void)epochLoss;
    (void)hasValLoss;
    (void)valLoss;
    (void)learningRate;

    std::cout << summary << "\n";
  }

  //===================================================================================================================//

  void CalibrateController::onTrainFinished(bool success, const std::string& finalSummary)
  {
    std::cout << "\n";

    std::string prefix = success ? "[Calibration complete] " : "[Calibration failed] ";
    std::cout << prefix << finalSummary << "\n";
  }

  //===================================================================================================================//

  void CalibrateController::onModelInfoUpdated(const std::string& property, const std::string& value)
  {
    std::cout << "  " << property << ": " << value << "\n";
  }

  //===================================================================================================================//

  void CalibrateController::onLogMessage(const std::string& message, bool isError)
  {
    if (isError)
      std::cerr << "[ERROR] " << message << "\n";
    else
      std::cout << message << "\n";
  }

  //===================================================================================================================//

  void CalibrateController::onTimingUpdated(const std::string& metric, float value)
  {
    std::cout << "  " << metric << ": " << std::fixed << std::setprecision(2) << value << " ms\n";
  }

  //===================================================================================================================//
  //-- Explicit template instantiations --//
  //===================================================================================================================//

} // namespace NN_CLI
