#ifndef ANN_CORE_H
#define ANN_CORE_H

#include "ANN_Types.hpp"
#include "ANN_CoreConfig.hpp"
#include "ANN_Sample.hpp"
#include "ANN_TrainingProgress.hpp"
#include "ANN_TrainingMetadata.hpp"
#include "ANN_PredictMetadata.hpp"
#include "ANN_TestResult.hpp"

#include <chrono>
#include <memory>

//==============================================================================//

namespace ANN {
  template <typename T>
  class Core {
    public:
      //-- Factory --//
      static std::unique_ptr<Core<T>> makeCore(const CoreConfig<T>& config);

      //-- Core interface --//
      virtual Output<T> predict(const Input<T>& input) = 0;
      virtual void train(const Samples<T>& samples) = 0;
      virtual TestResult<T> test(const Samples<T>& samples) = 0;

      //-- Step-by-step training (for external orchestration, e.g., CNN) --//
      // Usage flow: predict(input) → backpropagate(output) → [other layers backprop] → accumulate() → update(numSamples)
      virtual Tensor1D<T> backpropagate(const Output<T>& output) = 0;
      virtual void accumulate() = 0;
      virtual void resetAccumulators() = 0;
      virtual void update(ulong numSamples) = 0;

      //-- Getters --//
      ModeType getModeType() const { return modeType; }
      DeviceType getDeviceType() const { return deviceType; }
      int getNumThreads() const { return numThreads; }
      int getNumGPUs() const { return numGPUs; }
      const LayersConfig& getLayersConfig() const { return layersConfig; }
      const TrainingConfig<T>& getTrainingConfig() const { return trainingConfig; }
      const PredictMetadata<T>& getPredictMetadata() const { return predictMetadata; }
      const TrainingMetadata<T>& getTrainingMetadata() const { return trainingMetadata; }
      const Parameters<T>& getParameters() const { return parameters; }
      const CostFunctionConfig<T>& getCostFunctionConfig() const { return costFunctionConfig; }

      //-- Setters --//
      void setParameters(const Parameters<T>& params) { parameters = params; }
      void setTrainingCallback(TrainingCallback<T> callback) { trainingCallback = callback; }

      //-- Log level --//
      void setLogLevel(LogLevel level) { logLevel = level; }
      LogLevel getLogLevel() const { return logLevel; }

    protected:
      //-- Constructor / Validation --//
      explicit Core(const CoreConfig<T>& coreConfig);
      void sanityCheck(const CoreConfig<T>& coreConfig);

      //-- Loss calculation --//
      T calculateLoss(const Output<T>& expected);

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
      TrainingMetadata<T> trainingMetadata;
      PredictMetadata<T> predictMetadata;
      Parameters<T> parameters;
      CostFunctionConfig<T> costFunctionConfig;
      ulong progressReports = 1000;
      LogLevel logLevel = LogLevel::ERROR;

      //-- Internal state --//
      Tensor2D<T> actvs;
      Tensor2D<T> zs;

      TrainingCallback<T> trainingCallback;

    private:
      //-- Timing state --//
      std::chrono::time_point<std::chrono::system_clock> trainingStartTime;
      std::chrono::time_point<std::chrono::system_clock> predictStartTime;
  };
}

#endif // ANN_CORE_H
