#ifndef ANN_CORE_H
#define ANN_CORE_H

#include "ANN_ActvFunc.hpp"
#include "ANN_Mode.hpp"
#include "ANN_Device.hpp"
#include "ANN_LayersConfig.hpp"

#include <chrono>
#include <functional>
#include <memory>
#include <vector>

//==============================================================================//

namespace ANN {
  template <typename T>
  using Input = std::vector<T>;

  template <typename T>
  using Output = std::vector<T>;

  template <typename T>
  using Inputs = std::vector<Input<T>>;

  template <typename T>
  using Outputs = std::vector<Output<T>>;

  template <typename T>
  using Tensor1D = std::vector<T>;

  template <typename T>
  using Tensor2D = std::vector<std::vector<T>>;

  template <typename T>
  using Tensor3D = std::vector<std::vector<std::vector<T>>>;

  template <typename T>
  struct TrainingConfig {
    ulong numEpochs = 0;
    float learningRate = 0.01f;
    int numThreads = 0;           // 0 = use all available cores (for CPU mode)
    int numGPUs = 0;              // 0 = use all available GPUs, 1 = single GPU (default behavior)
    ulong progressReports = 1000; // Number of progress reports per epoch (0 = no reports, default = 1000)
  };

  template <typename T>
  struct Parameters {
    Tensor3D<T> weights;
    Tensor2D<T> biases;
  };

  template <typename T>
  struct CoreConfig {
    ModeType modeType;
    DeviceType deviceType;
    LayersConfig layersConfig;
    TrainingConfig<T> trainingConfig;
    Parameters<T> parameters;
  };

  template <typename T>
  struct Sample {
    Input<T> input;
    Output<T> output;
  };

  template <typename T>
  using Samples = std::vector<Sample<T>>;

  // Training progress information passed to callbacks
  template <typename T>
  struct TrainingProgress {
    ulong currentEpoch;
    ulong totalEpochs;
    ulong currentSample;
    ulong totalSamples;
    T epochLoss;        // Average loss for completed epoch (0 if epoch not complete)
    T sampleLoss;       // Loss for current sample

    // Multi-GPU progress tracking
    int gpuIndex = -1;  // -1 = not GPU-specific (epoch completion, CPU mode), >= 0 = specific GPU
    int totalGPUs = 1;  // Total number of GPUs being used
  };

  // Callback type for training progress
  template <typename T>
  using TrainingCallback = std::function<void(const TrainingProgress<T>&)>;

  // Training metadata (captured at runtime, saved with the model)
  template <typename T>
  struct TrainingMetadata {
    std::string startTime;      // ISO 8601 format (e.g., "2026-02-19T10:30:00")
    std::string endTime;        // ISO 8601 format
    double durationSeconds;     // Total training duration in seconds
    std::string durationFormatted; // Human-readable duration (e.g., "1y 2mo 3d 4h 5m 6s")
    ulong numSamples;           // Number of training samples used
    T finalLoss;                // Average loss at the end of training
  };

  // Test result information
  template <typename T>
  struct TestResult {
    ulong numSamples;     // Total number of samples tested
    T totalLoss;          // Sum of all sample losses
    T averageLoss;        // Average loss per sample (totalLoss / numSamples)
  };

  template <typename T>
  class Core {
    public:
      static std::unique_ptr<Core<T>> makeCore(const CoreConfig<T>& config);

      virtual Output<T> run(const Input<T>& input) = 0;
      virtual void train(const Samples<T>& samples) = 0;
      virtual TestResult<T> test(const Samples<T>& samples) = 0;

      ModeType getModeType() const { return modeType; }
      DeviceType getDeviceType() const { return deviceType; }
      const LayersConfig& getLayersConfig() const { return layersConfig; }
      const TrainingConfig<T>& getTrainingConfig() const { return trainingConfig; }
      const TrainingMetadata<T>& getTrainingMetadata() const { return trainingMetadata; }
      const Parameters<T>& getParameters() const { return parameters; }

      // Set parameters (for multi-GPU weight synchronization)
      void setParameters(const Parameters<T>& params) { parameters = params; }

      // Set a callback to receive training progress updates
      void setTrainingCallback(TrainingCallback<T> callback) { trainingCallback = callback; }

    protected:
      explicit Core(const CoreConfig<T>& coreConfig);
      void sanityCheck(const CoreConfig<T>& coreConfig);

      // Calculate MSE loss between output activations and expected output
      T calculateLoss(const Output<T>& expected);

      // Training timing helpers - called at start/end of training
      void trainingStart(ulong numSamples);
      TrainingMetadata<T> trainingEnd();

      DeviceType deviceType;
      ModeType modeType;
      LayersConfig layersConfig;
      TrainingConfig<T> trainingConfig;
      TrainingMetadata<T> trainingMetadata;
      Parameters<T> parameters;

      Tensor2D<T> actvs;
      Tensor2D<T> zs;

      TrainingCallback<T> trainingCallback;

    private:
      std::chrono::time_point<std::chrono::system_clock> trainingStartTime;
  };
}

#endif // ANN_CORE_H
