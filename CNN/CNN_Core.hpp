#ifndef CNN_CORE_HPP
#define CNN_CORE_HPP

#include "CNN_Types.hpp"
#include "CNN_CoreConfig.hpp"
#include "CNN_InputProvider.hpp"
#include "CNN_Sample.hpp"
#include "CNN_SampleProvider.hpp"
#include "Common/Common_EpochRecord.hpp"
#include "Common/Common_TrainingProgressEvent.hpp"
#include "Common/Common_TrainMetadata.hpp"
#include "Common/Common_PredictMetadata.hpp"
#include "Common/Common_PredictResult.hpp"
#include "Common/Common_ProgressCallback.hpp"
#include "Common/Common_TestResult.hpp"
#include "CNN_TimingCallback.hpp"

#include <atomic>
#include <chrono>
#include <memory>

//===================================================================================================================//

namespace CNN
{
  using namespace Common;
  template <typename T>
  class Core
  {
    public:
      //-- Factory --//
      static std::unique_ptr<Core<T>> makeCore(const CoreConfig<T>& config);

      //-- Core interface --//
      // Streaming predict: pulls inputs in batches from `provider`. Lets
      // callers score arbitrarily large datasets without holding the full
      // Inputs<T> in memory. Eager overloads below wrap this one.
      virtual Common::PredictResults<T> predict(ulong numSamples, const InputProvider<T>& provider) = 0;
      // Eager overloads (concrete wrappers around the streaming version).
      // Each result carries .output (post-activation, e.g. softmax probs)
      // and .logits (pre-activation z of the dense head's last layer, used
      // for calibration / OOD scores that softmax discards).
      Common::PredictResults<T> predict(const Inputs<T>& inputs);
      Common::PredictResult<T> predict(const Input<T>& input);
      virtual void train(ulong numSamples, const SampleProvider<T>& sampleProvider) = 0;
      virtual Common::TestResult<T> test(ulong numSamples, const SampleProvider<T>& sampleProvider) = 0;

      //-- Destructor --//
      virtual ~Core() = default;

      //-- Getters --//
      Common::ModeType getModeType() const
      {
        return modeType;
      }

      Common::DeviceType getDeviceType() const
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

      const Shape3D& getInputShape() const
      {
        return inputShape;
      }

      const LayersConfig& getLayersConfig() const
      {
        return layersConfig;
      }

      const Common::TrainConfig<T>& getTrainConfig() const
      {
        return trainConfig;
      }

      const Common::PredictMetadata<T>& getPredictMetadata() const
      {
        return predictMetadata;
      }

      const Common::TrainMetadata<T>& getTrainMetadata() const
      {
        return trainMetadata;
      }

      Common::TrainMetadata<T>& getTrainMetadata()
      {
        return trainMetadata;
      }

      const Parameters<T>& getParameters() const
      {
        return parameters;
      }

      const Common::CostFunctionConfig<T>& getCostFunctionConfig() const
      {
        return coreConfig.costFunctionConfig;
      }

      //-- Setters --//
      void setParameters(const Parameters<T>& params)
      {
        parameters = params;
      }

      // Upload current host-side parameters to GPU buffers (no-op for CPU cores).
      virtual void syncParametersToGPU() {}

      void setTrainingCallback(Common::TrainingCallback<T> callback)
      {
        trainingCallback = callback;
      }

      // Invoked once per completed epoch with the 0-based epoch index. The
      // consumer performs epoch-boundary work here (validation, checkpoints).
      void setEpochCompletedCallback(Common::EpochCompletedCallback<T> callback)
      {
        epochCompletedCallback = callback;
      }

      void setProgressCallback(Common::ProgressCallback callback)
      {
        progressCallback = callback;
      }

      // Optional instrumentation hook. When set, the training loop notifies the
      // consumer at phase boundaries (begin/end) so it can measure durations.
      // No-op overhead when unset.
      void setTimingCallback(TimingCallback callback)
      {
        timingCallback = callback;
      }

      void setGpuProfileCallback(GpuProfileCallback callback)
      {
        gpuProfileCallback = callback;
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
      void setLogLevel(Common::LogLevel level)
      {
        logLevel = level;
      }

      Common::LogLevel getLogLevel() const
      {
        return logLevel;
      }

      //-- Epoch history --//
      // Prepend loaded epoch history (from a resumed model) before the
      // newly-trained epochs so the serializer writes the full history.
      void prependEpochHistory(const std::vector<Common::EpochRecord<T>>& history)
      {
        std::vector<Common::EpochRecord<T>> merged = history;
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
      Common::TrainMetadata<T> trainingEnd();

      //-- Predict timing --//
      void predictStart();
      Common::PredictMetadata<T> predictEnd();

      //-- Configuration members --//
      CoreConfig<T> coreConfig;
      Common::DeviceType deviceType;
      Common::ModeType modeType;
      int numThreads = 0;
      int numGPUs = 0;
      Shape3D inputShape;
      LayersConfig layersConfig;
      Common::TrainConfig<T> trainConfig;
      Common::TestConfig testConfig;
      Common::TrainMetadata<T> trainMetadata;
      Common::PredictMetadata<T> predictMetadata;
      Parameters<T> parameters;
      ulong progressReports = 1000;
      Common::LogLevel logLevel = Common::LogLevel::ERROR;

      //-- Internal state --//
      TrainingCallback<T> trainingCallback;
      EpochCompletedCallback<T> epochCompletedCallback;
      ProgressCallback progressCallback;
      TimingCallback timingCallback;
      GpuProfileCallback gpuProfileCallback;
      std::atomic<bool> stopRequested{false};

      // Notify the consumer that a measurable phase begins/ends. Cheap no-op when unset.
      void emitTiming(TimingPhase phase, TimingEvent event, int gpuIndex = -1) const
      {
        if (timingCallback)
          timingCallback(phase, event, gpuIndex);
      }

    private:
      //-- Timing state --//
      std::chrono::time_point<std::chrono::system_clock> trainingStartTime;
      std::chrono::time_point<std::chrono::system_clock> predictStartTime;
  };
}

//===================================================================================================================//

#endif // CNN_CORE_HPP
