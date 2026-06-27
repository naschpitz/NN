#include "NN-CLI_CalibrateController.hpp"

#include "NN-CLI_ANNRunner.hpp"
#include "NN-CLI_CNNRunner.hpp"

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace NN_CLI
{

  //===================================================================================================================//
  //-- Ctors / Dtors --//
  //===================================================================================================================//

  template <typename RunnerT>
  CalibrateController<RunnerT>::~CalibrateController()
  {
    // Signal connections auto-disconnect when runnerSignals (sender) is destroyed.
  }

  //===================================================================================================================//
  //-- Lifecycle --//
  //===================================================================================================================//

  template <typename RunnerT>
  void CalibrateController<RunnerT>::init(std::unique_ptr<RunnerT> runner)
  {
    this->runner = std::move(runner);

    if (!this->runner)
      return;

    // Calibration runs synchronously with no event loop.  DirectConnection
    // delivers the slots inline on the emitting thread (same threading
    // semantics as the original observer pattern), so no queued events pile up
    // while the main thread is blocked in runner->calibrate().
    auto& hub = this->runner->getRunnerSignals();
    auto* ctx = &this->signalContext;

    QObject::connect(
      &hub, &RunnerSignals::batchProgress, ctx,
      [this](int batchIdx, int totalBatches, float currentLoss, float samplesPerSec, float etaSeconds,
             const std::vector<float>& fractions) {
        this->onBatchProgress(batchIdx, totalBatches, currentLoss, samplesPerSec, etaSeconds, fractions);
      },

      Qt::DirectConnection);

    QObject::connect(
      &hub, &RunnerSignals::epochCompleted, ctx,
      [this](int epochIdx, int totalEpochs, float epochLoss, bool hasValLoss, float valLoss, float learningRate,
             const std::string& summary) {
        this->onEpochCompleted(epochIdx, totalEpochs, epochLoss, hasValLoss, valLoss, learningRate, summary);
      },

      Qt::DirectConnection);

    QObject::connect(
      &hub, &RunnerSignals::trainFinished, ctx,
      [this](bool success, const std::string& finalSummary) { this->onTrainFinished(success, finalSummary); },
      Qt::DirectConnection);

    QObject::connect(
      &hub, &RunnerSignals::modelInfoUpdated, ctx,
      [this](const std::string& property, const std::string& value) { this->onModelInfoUpdated(property, value); },
      Qt::DirectConnection);

    QObject::connect(
      &hub, &RunnerSignals::logMessage, ctx,
      [this](const std::string& message, bool isError) { this->onLogMessage(message, isError); }, Qt::DirectConnection);

    QObject::connect(
      &hub, &RunnerSignals::timingUpdated, ctx,
      [this](const std::string& metric, float value) { this->onTimingUpdated(metric, value); }, Qt::DirectConnection);
  }

  //===================================================================================================================//

  template <typename RunnerT>
  int CalibrateController<RunnerT>::startCalibrate()
  {
    if (!this->runner)
      return 1;

    return this->runner->calibrate();
  }

  //===================================================================================================================//
  //-- Accessors --//
  //===================================================================================================================//

  template <typename RunnerT>
  RunnerT* CalibrateController<RunnerT>::getRunner() const
  {
    return this->runner.get();
  }

  //===================================================================================================================//
  //-- Runner signal handlers --//
  //===================================================================================================================//

  template <typename RunnerT>
  void CalibrateController<RunnerT>::onBatchProgress(int batchIdx, int totalBatches, float currentLoss,
                                                     float samplesPerSec, float etaSeconds,
                                                     const std::vector<float>& fractions)
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

  template <typename RunnerT>
  void CalibrateController<RunnerT>::onEpochCompleted(int epochIdx, int totalEpochs, float epochLoss, bool hasValLoss,
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

  template <typename RunnerT>
  void CalibrateController<RunnerT>::onTrainFinished(bool success, const std::string& finalSummary)
  {
    std::cout << "\n";

    std::string prefix = success ? "[Calibration complete] " : "[Calibration failed] ";
    std::cout << prefix << finalSummary << "\n";
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void CalibrateController<RunnerT>::onModelInfoUpdated(const std::string& property, const std::string& value)
  {
    std::cout << "  " << property << ": " << value << "\n";
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void CalibrateController<RunnerT>::onLogMessage(const std::string& message, bool isError)
  {
    if (isError)
      std::cerr << "[ERROR] " << message << "\n";
    else
      std::cout << message << "\n";
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void CalibrateController<RunnerT>::onTimingUpdated(const std::string& metric, float value)
  {
    std::cout << "  " << metric << ": " << std::fixed << std::setprecision(2) << value << " ms\n";
  }

  //===================================================================================================================//
  //-- Explicit template instantiations --//
  //===================================================================================================================//

  template class CalibrateController<ANNRunner>;
  template class CalibrateController<CNNRunner>;

} // namespace NN_CLI
