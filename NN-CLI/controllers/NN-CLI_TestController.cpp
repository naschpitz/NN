#include "NN-CLI_TestController.hpp"

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace NN_CLI
{

  //===================================================================================================================//
  //-- Ctors / Dtors --//
  //===================================================================================================================//

  TestController::~TestController()
  {
    // Signal connections auto-disconnect when the Runner (sender) is destroyed.
  }

  //===================================================================================================================//
  //-- Lifecycle --//
  //===================================================================================================================//

  void TestController::init(std::unique_ptr<RunnerBase> runner)
  {
    this->runner = std::move(runner);

    if (!this->runner)
      return;

    // Test runs synchronously with no event loop.  DirectConnection delivers
    // the slots inline on the emitting thread (same threading semantics as the
    // original observer pattern), so no queued events pile up while the main
    // thread is blocked in runner->test().
    QObject::connect(this->runner.get(), &RunnerBase::batchProgress, this, &TestController::onBatchProgress,
                     Qt::DirectConnection);
    QObject::connect(this->runner.get(), &RunnerBase::epochCompleted, this, &TestController::onEpochCompleted,
                     Qt::DirectConnection);
    QObject::connect(this->runner.get(), &RunnerBase::trainFinished, this, &TestController::onTrainFinished,
                     Qt::DirectConnection);
    QObject::connect(this->runner.get(), &RunnerBase::modelInfoUpdated, this, &TestController::onModelInfoUpdated,
                     Qt::DirectConnection);
    QObject::connect(this->runner.get(), &RunnerBase::logMessage, this, &TestController::onLogMessage,
                     Qt::DirectConnection);
    QObject::connect(this->runner.get(), &RunnerBase::timingUpdated, this, &TestController::onTimingUpdated,
                     Qt::DirectConnection);
  }

  //===================================================================================================================//

  int TestController::startTest()
  {
    if (!this->runner)
      return 1;

    return this->runner->test();
  }

  //===================================================================================================================//
  //-- Accessors --//
  //===================================================================================================================//

  RunnerBase* TestController::getRunner() const
  {
    return this->runner.get();
  }

  //===================================================================================================================//
  //-- Runner signal handlers --//
  //===================================================================================================================//

  void TestController::onBatchProgress(int batchIdx, int totalBatches, float currentLoss, float samplesPerSec,
                                       float etaSeconds, const std::vector<float>& fractions)
  {
    // Test mode: batch progress maps to batch evaluation progress with loss.
    (void)samplesPerSec;
    (void)etaSeconds;

    float fraction = fractions.empty() ? 0.0f : fractions[0];
    std::cout << "\r  Progress: " << (batchIdx + 1) << "/" << totalBatches << " (" << std::fixed << std::setprecision(1)
              << (fraction * 100.0f) << "%)" << "  Loss: " << std::fixed << std::setprecision(6) << currentLoss
              << std::flush;
  }

  //===================================================================================================================//

  void TestController::onEpochCompleted(int epochIdx, int totalEpochs, float epochLoss, bool hasValLoss, float valLoss,
                                        float learningRate, const std::string& summary)
  {
    // Test mode does not use epoch events, but print the summary for
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

  void TestController::onTrainFinished(bool success, const std::string& finalSummary)
  {
    std::cout << "\n";

    std::string prefix = success ? "[Test complete] " : "[Test failed] ";
    std::cout << prefix << finalSummary << "\n";
  }

  //===================================================================================================================//

  void TestController::onModelInfoUpdated(const std::string& property, const std::string& value)
  {
    std::cout << "  " << property << ": " << value << "\n";
  }

  //===================================================================================================================//

  void TestController::onLogMessage(const std::string& message, bool isError)
  {
    if (isError)
      std::cerr << "[ERROR] " << message << "\n";
    else
      std::cout << message << "\n";
  }

  //===================================================================================================================//

  void TestController::onTimingUpdated(const std::string& metric, float value)
  {
    std::cout << "  " << metric << ": " << std::fixed << std::setprecision(2) << value << " ms\n";
  }

  //===================================================================================================================//
  //-- Explicit template instantiations --//
  //===================================================================================================================//

} // namespace NN_CLI
