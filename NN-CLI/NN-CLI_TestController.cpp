#include "NN-CLI_TestController.hpp"

#include "NN-CLI_ANNRunner.hpp"
#include "NN-CLI_CNNRunner.hpp"

#include <iomanip>
#include <iostream>
#include <string>

namespace NN_CLI
{

  //===================================================================================================================//
  //-- Ctors / Dtors --//
  //===================================================================================================================//

  template <typename RunnerT>
  TestController<RunnerT>::~TestController()
  {
    if (this->runner)
      this->runner->removeObserver(this);
  }

  //===================================================================================================================//
  //-- Lifecycle --//
  //===================================================================================================================//

  template <typename RunnerT>
  void TestController<RunnerT>::init(std::unique_ptr<RunnerT> runner)
  {
    this->runner = std::move(runner);

    if (this->runner)
      this->runner->addObserver(this);
  }

  //===================================================================================================================//

  template <typename RunnerT>
  int TestController<RunnerT>::startTest()
  {
    if (!this->runner)
      return 1;

    return this->runner->test();
  }

  //===================================================================================================================//
  //-- Accessors --//
  //===================================================================================================================//

  template <typename RunnerT>
  RunnerT* TestController<RunnerT>::getRunner() const
  {
    return this->runner.get();
  }

  //===================================================================================================================//
  //-- IRunnerObserver overrides --//
  //===================================================================================================================//

  template <typename RunnerT>
  void TestController<RunnerT>::onBatchProgress(int batchIdx, int totalBatches, float currentLoss, float fraction)
  {
    // Test mode: batch progress maps to batch evaluation progress with loss.
    std::cout << "\r  Progress: " << (batchIdx + 1) << "/" << totalBatches << " ("
              << std::fixed << std::setprecision(1) << (fraction * 100.0f) << "%)"
              << "  Loss: " << std::fixed << std::setprecision(6) << currentLoss << std::flush;
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TestController<RunnerT>::onEpochCompleted(int epochIdx, int totalEpochs, float epochLoss, float accuracy,
                                                  const std::string& summary)
  {
    // Test mode does not use epoch events, but print the summary for
    // interface completeness in case the runner fires one.
    (void)epochIdx;
    (void)totalEpochs;
    (void)epochLoss;
    (void)accuracy;

    std::cout << summary << "\n";
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TestController<RunnerT>::onTrainingFinished(bool success, const std::string& finalSummary)
  {
    std::cout << "\n";

    std::string prefix = success ? "[Test complete] " : "[Test failed] ";
    std::cout << prefix << finalSummary << "\n";
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TestController<RunnerT>::onModelInfoUpdated(const std::string& property, const std::string& value)
  {
    std::cout << "  " << property << ": " << value << "\n";
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TestController<RunnerT>::onLogMessage(const std::string& message, bool isError)
  {
    if (isError)
      std::cerr << "[ERROR] " << message << "\n";
    else
      std::cout << message << "\n";
  }

  //===================================================================================================================//

  template <typename RunnerT>
  void TestController<RunnerT>::onTimingUpdated(const std::string& metric, float value)
  {
    std::cout << "  " << metric << ": " << std::fixed << std::setprecision(2) << value << " ms\n";
  }

  //===================================================================================================================//
  //-- Explicit template instantiations --//
  //===================================================================================================================//

  template class TestController<ANNRunner>;
  template class TestController<CNNRunner>;

} // namespace NN_CLI
