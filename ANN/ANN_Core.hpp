#ifndef ANN_CORE_H
#define ANN_CORE_H

#include "ANN_Types.hpp"
#include "ANN_CoreConfig.hpp"
#include "ANN_Sample.hpp"
#include "ANN_TrainingProgress.hpp"
#include "ANN_TrainingMetadata.hpp"
#include "ANN_PredictMetadata.hpp"
#include "ANN_PredictResult.hpp"
#include "ANN_ProgressCallback.hpp"
#include "ANN_TestResult.hpp"

#include <atomic>
#include <chrono>
#include <memory>

//==============================================================================//

namespace ANN
{
  template <typename T>
  class Core
  {
  public:
    //-- Factory --//
    static std::unique_ptr<Core<T>> makeCore(const CoreConfig<T>& config);

    //-- Core interface --//
    // predict returns both the post-activation output and the pre-activation (z) of the
    // last layer. The logits are required for calibration / OOD-detection scores
    // (max-logit, logit-norm, free-energy) that softmax discards.
    virtual PredictResults<T> predict(const Inputs<T>& inputs) = 0;
    virtual PredictResult<T> predict(const Input<T>& input); // Overridden in CoreCPU to avoid threading
    virtual void train(ulong numSamples, const SampleProvider<T>& sampleProvider) = 0;
    virtual TestResult<T> test(ulong numSamples, const SampleProvider<T>& sampleProvider) = 0;

    //-- Step-by-step training (for external orchestration) --//
    // Used by external orchestrators that embed ANN as a sub-network (e.g., a dense
    // head after convolutional layers) and need the input-layer gradients to continue
    // backpropagation through their own preceding layers.
    // Usage: predict(input) → backpropagate(expected) → accumulate() → update(numSamples)
    virtual Tensor1D<T> backpropagate(const Output<T>& expected) = 0;
    virtual void accumulate() = 0;
    virtual void resetAccumulators() = 0;
    virtual void update(ulong numSamples) = 0;

    //-- Getters --//
    ModeType getModeType() const
    {
      return modeType;
    }

    DeviceType getDeviceType() const
    {
      return deviceType;
    }

    int getNumThreads() const
    {
      return numThreads;
    }

    int getNumGPUs() const
    {
      return numGPUs;
    }

    const LayersConfig& getLayersConfig() const
    {
      return layersConfig;
    }

    const TrainingConfig<T>& getTrainingConfig() const
    {
      return trainingConfig;
    }

    const PredictMetadata<T>& getPredictMetadata() const
    {
      return predictMetadata;
    }

    const TrainingMetadata<T>& getTrainingMetadata() const
    {
      return trainingMetadata;
    }

    const Parameters<T>& getParameters() const
    {
      return parameters;
    }

    const CostFunctionConfig<T>& getCostFunctionConfig() const
    {
      return costFunctionConfig;
    }

    //-- Setters --//
    void setParameters(const Parameters<T>& params)
    {
      parameters = params;
    }

    void setTrainingCallback(TrainingCallback<T> callback)
    {
      trainingCallback = callback;
    }

    void setProgressCallback(ProgressCallback callback)
    {
      progressCallback = callback;
    }

    // Request early training termination. The training loop checks this at epoch boundaries.
    void requestStop()
    {
      stopRequested.store(true);
    }

    bool isStopRequested() const
    {
      return stopRequested.load();
    }

    //-- Log level --//
    void setLogLevel(LogLevel level)
    {
      logLevel = level;
    }

    LogLevel getLogLevel() const
    {
      return logLevel;
    }

  protected:
    //-- Constructor / Validation --//
    explicit Core(const CoreConfig<T>& coreConfig);
    void sanityCheck(const CoreConfig<T>& coreConfig);

    //-- Training timing --//
    void trainingStart(ulong numSamples);
    TrainingMetadata<T> trainingEnd();

    //-- Predict timing --//
    void predictStart();
    PredictMetadata<T> predictEnd();

    //-- Configuration members --//
    DeviceType deviceType;
    ModeType modeType;
    int numThreads = 0;
    int numGPUs = 0;
    LayersConfig layersConfig;
    TrainingConfig<T> trainingConfig;
    TestConfig testConfig;
    TrainingMetadata<T> trainingMetadata;
    PredictMetadata<T> predictMetadata;
    Parameters<T> parameters;
    CostFunctionConfig<T> costFunctionConfig;
    ulong progressReports = 1000;
    LogLevel logLevel = LogLevel::ERROR;

    TrainingCallback<T> trainingCallback;
    ProgressCallback progressCallback;
    std::atomic<bool> stopRequested{false};

  private:
    //-- Timing state --//
    std::chrono::time_point<std::chrono::system_clock> trainingStartTime;
    std::chrono::time_point<std::chrono::system_clock> predictStartTime;
  };
}

#endif // ANN_CORE_H
