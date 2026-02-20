#ifndef ANN_CORE_H
#define ANN_CORE_H

#include "ANN_Types.hpp"
#include "ANN_CoreConfig.hpp"
#include "ANN_Sample.hpp"
#include "ANN_TrainingProgress.hpp"
#include "ANN_TrainingMetadata.hpp"
#include "ANN_RunMetadata.hpp"
#include "ANN_TestResult.hpp"

#include <chrono>
#include <memory>

//==============================================================================//

namespace ANN {
  template <typename T>
  class Core {
    public:
      static std::unique_ptr<Core<T>> makeCore(const CoreConfig<T>& config);

      virtual Output<T> inference(const Input<T>& input) = 0;
      virtual void train(const Samples<T>& samples) = 0;
      virtual TestResult<T> test(const Samples<T>& samples) = 0;

      ModeType getModeType() const { return modeType; }
      DeviceType getDeviceType() const { return deviceType; }
      const LayersConfig& getLayersConfig() const { return layersConfig; }
      const TrainingConfig<T>& getTrainingConfig() const { return trainingConfig; }
      const RunMetadata<T>& getInferenceMetadata() const { return inferenceMetadata; }
      const TrainingMetadata<T>& getTrainingMetadata() const { return trainingMetadata; }
      const Parameters<T>& getParameters() const { return parameters; }

      // Set parameters (for multi-GPU weight synchronization)
      void setParameters(const Parameters<T>& params) { parameters = params; }

      // Set a callback to receive training progress updates
      void setTrainingCallback(TrainingCallback<T> callback) { trainingCallback = callback; }

      // Verbose control
      void setVerbose(bool v) { verbose = v; }
      bool isVerbose() const { return verbose; }

    protected:
      explicit Core(const CoreConfig<T>& coreConfig);
      void sanityCheck(const CoreConfig<T>& coreConfig);

      // Calculate MSE loss between output activations and expected output
      T calculateLoss(const Output<T>& expected);

      // Training timing helpers - called at start/end of training
      void trainingStart(ulong numSamples);
      TrainingMetadata<T> trainingEnd();

      // Inference timing helpers - called at start/end of inference
      void inferenceStart();
      RunMetadata<T> inferenceEnd();

      DeviceType deviceType;
      ModeType modeType;
      LayersConfig layersConfig;
      TrainingConfig<T> trainingConfig;
      TrainingMetadata<T> trainingMetadata;
      RunMetadata<T> inferenceMetadata;
      Parameters<T> parameters;
      bool verbose = false;

      Tensor2D<T> actvs;
      Tensor2D<T> zs;

      TrainingCallback<T> trainingCallback;

    private:
      std::chrono::time_point<std::chrono::system_clock> trainingStartTime;
      std::chrono::time_point<std::chrono::system_clock> inferenceStartTime;
  };
}

#endif // ANN_CORE_H
