#include "NN-CLI_PredictController.hpp"

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
  PredictController<RunnerT>::~PredictController()
  {
    if (this->runner)
      this->runner->removeObserver(this);
  }

  //===================================================================================================================//
  //-- Lifecycle --//
  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::init(std::unique_ptr<RunnerT> runner)
  {
    this->runner = std::move(runner);

    if (this->runner)
      this->runner->addObserver(this);
  }

  //===================================================================================================================//

  template <typename RunnerT>
  int PredictController<RunnerT>::startPredict()
  {
    if (!this->runner)
      return 1;

    return this->runner->predict();
  }

  //===================================================================================================================//
  //-- Accessors --//
  //===================================================================================================================//

  template <typename RunnerT>
  RunnerT* PredictController<RunnerT>::getRunner() const
  {
    return this->runner.get();
  }

  //===================================================================================================================//
  //-- IRunnerObserver overrides --//
  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::onBatchProgress(int batchIdx, int totalBatches, float currentLoss,
                                                      float samplesPerSec, float etaSeconds,
                                                      const std::vector<float>& fractions)
  {
    // Predict mode: batch progress maps to sample processing progress.
    (void)currentLoss;
    (void)samplesPerSec;
    (void)etaSeconds;

    float fraction = fractions.empty() ? 0.0f : fractions[0];
    std::cout << "\r  Progress: " << (batchIdx + 1) << "/" << totalBatches << " ("
              << std::fixed << std::setprecision(1) << (fraction * 100.0f) << "%)" << std::flush;
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::onEpochCompleted(int epochIdx, int totalEpochs, float epochLoss, bool hasValLoss,
                                                     float valLoss, const std::string& summary)
  {
    // Predict mode does not use epoch events, but print the summary for
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
  void PredictController<RunnerT>::onTrainingFinished(bool success, const std::string& finalSummary)
  {
    std::cout << "\n";

    std::string prefix = success ? "[Predict complete] " : "[Predict failed] ";
    std::cout << prefix << finalSummary << "\n";
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::onModelInfoUpdated(const std::string& property, const std::string& value)
  {
    std::cout << "  " << property << ": " << value << "\n";
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::onLogMessage(const std::string& message, bool isError)
  {
    if (isError)
      std::cerr << "[ERROR] " << message << "\n";
    else
      std::cout << message << "\n";
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void PredictController<RunnerT>::onTimingUpdated(const std::string& metric, float value)
  {
    std::cout << "  " << metric << ": " << std::fixed << std::setprecision(2) << value << " ms\n";
  }

  //===================================================================================================================//
  //-- Explicit template instantiations --//
  //===================================================================================================================//

  template class PredictController<ANNRunner>;
  template class PredictController<CNNRunner>;

} // namespace NN_CLI
