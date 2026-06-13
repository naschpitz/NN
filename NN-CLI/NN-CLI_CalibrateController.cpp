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
    if (this->runner)
      this->runner->removeObserver(this);
  }

  //===================================================================================================================//
  //-- Lifecycle --//
  //===================================================================================================================//

  template <typename RunnerT>
  void CalibrateController<RunnerT>::init(std::unique_ptr<RunnerT> runner)
  {
    this->runner = std::move(runner);

    if (this->runner)
      this->runner->addObserver(this);
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
  //-- IRunnerObserver overrides --//
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
                                                      float valLoss, const std::string& summary)
  {
    // Calibrate mode does not use epoch events, but print the summary for
    // interface completeness in case the runner fires one.
    (void)epochIdx;
    (void)totalEpochs;
    (void)epochLoss;
    (void)hasValLoss;
    (void)valLoss;

    std::cout << summary << "\n";
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void CalibrateController<RunnerT>::onTrainingFinished(bool success, const std::string& finalSummary)
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
