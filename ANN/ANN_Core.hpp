#ifndef ANN_CORE_H
#define ANN_CORE_H

#include "ANN_Types.hpp"
#include "ANN_CoreConfig.hpp"
#include "ANN_InputProvider.hpp"
#include "ANN_Sample.hpp"
#include "ANN_SampleProvider.hpp"
#include "Common/Common_EpochRecord.hpp"
#include "Common/Common_TrainingProgressEvent.hpp"
#include "Common/Common_TrainMetadata.hpp"
#include "Common/Common_PredictMetadata.hpp"
#include "Common/Common_PredictResult.hpp"
#include "Common/Common_ProgressCallback.hpp"
#include "Common/Common_TestResult.hpp"

#include <atomic>
#include <chrono>
#include <memory>

//==============================================================================//

namespace ANN
{
  using namespace Common;
  template <typename T>
  class Core
  {
    public:
      //-- Factory --//
      static std::unique_ptr<Core<T>> makeCore(const CoreConfig<T>& config);

      //-- Core interface --//
      // Streaming predict: pulls inputs in batches from `provider` and runs
      // forward propagation on each batch. Callers with an in-memory Inputs<T>
      // build a one-shot provider over it; callers with images / IDX files
      // decode each batch lazily. .output is the post-activation, .logits is
      // the pre-activation z of the last layer (used for calibration /
      // OOD scores that softmax discards).
      virtual PredictResults<T> predict(ulong numSamples, const InputProvider<T>& provider) = 0;
      // Single-input convenience. Overridden in CoreCPU to avoid spawning a
      // thread pool on a single sample.
      virtual PredictResult<T> predict(const Input<T>& input);
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

      const TrainConfig<T>& getTrainConfig() const
      {
        return trainConfig;
      }

      const PredictMetadata<T>& getPredictMetadata() const
      {
        return predictMetadata;
      }

      const TrainMetadata<T>& getTrainMetadata() const
      {
        return trainMetadata;
      }

      TrainMetadata<T>& getTrainMetadata()
      {
        return trainMetadata;
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

      // Invoked once per completed epoch with the 0-based epoch index. The
      // consumer performs epoch-boundary work here (validation, checkpoints).
      void setEpochCompletedCallback(EpochCompletedCallback<T> callback)
      {
        epochCompletedCallback = callback;
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

      //-- Epoch history --//
      // Prepend loaded epoch history (from a resumed model) before the
      // newly-trained epochs so the serializer writes the full history.
      void prependEpochHistory(const std::vector<EpochRecord<T>>& history)
      {
        std::vector<EpochRecord<T>> merged = history;
        merged.insert(merged.end(), this->trainMetadata.epochHistory.begin(),
                      this->trainMetadata.epochHistory.end());
        this->trainMetadata.epochHistory = std::move(merged);
      }

    protected:
      //-- Constructor / Validation --//
      explicit Core(const CoreConfig<T>& coreConfig);
      void sanityCheck(const CoreConfig<T>& coreConfig);

      //-- Training timing --//
      void trainingStart(ulong numSamples);
       TrainMetadata<T> trainingEnd();

      //-- Predict timing --//
      void predictStart();
      PredictMetadata<T> predictEnd();

      //-- Configuration members --//
      DeviceType deviceType;
      ModeType modeType;
      int numThreads = 0;
      int numGPUs = 0;
      LayersConfig layersConfig;
      TrainConfig<T> trainConfig;
      TestConfig testConfig;
      TrainMetadata<T> trainMetadata;
      PredictMetadata<T> predictMetadata;
      Parameters<T> parameters;
      CostFunctionConfig<T> costFunctionConfig;
      ulong progressReports = 1000;
      LogLevel logLevel = LogLevel::ERROR;

      TrainingCallback<T> trainingCallback;
      EpochCompletedCallback<T> epochCompletedCallback;
      ProgressCallback progressCallback;
      std::atomic<bool> stopRequested{false};

    private:
      //-- Timing state --//
      std::chrono::time_point<std::chrono::system_clock> trainingStartTime;
      std::chrono::time_point<std::chrono::system_clock> predictStartTime;
  };
}

#endif // ANN_CORE_H
