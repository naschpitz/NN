#ifndef ANN_COREGPUWORKER_H
#define ANN_COREGPUWORKER_H

#include "ANN_Core.hpp"

#include <OCLW_Core.hpp>

#include <memory>

//===================================================================================================================//

namespace ANN {
  template <typename T>
  class CoreGPUWorker {
    public:
      // Standalone constructor — creates its own OpenCL core
      CoreGPUWorker(const LayersConfig& layersConfig, const TrainingConfig<T>& trainingConfig,
                    const Parameters<T>& parameters, ulong progressReports = 1000,
                    LogLevel logLevel = LogLevel::ERROR);

      // Shared-core constructor — uses externally-provided OpenCL core (for CNN integration).
      // Only initializes parameters. Caller must invoke loadSources() and allocateBuffers() manually.
      CoreGPUWorker(const LayersConfig& layersConfig, const TrainingConfig<T>& trainingConfig,
                    const Parameters<T>& parameters, OpenCLWrapper::Core& sharedCore,
                    ulong progressReports = 1000,
                    LogLevel logLevel = LogLevel::ERROR);

      //-- Predict --//
      Output<T> predict(const Input<T>& input);

      //-- Training (called by CoreGPU orchestrator) --//
      T trainSubset(const Samples<T>& samples, ulong startIdx, ulong endIdx, ulong epoch, ulong totalEpochs,
                    const TrainingCallback<T>& callback);

      //-- Testing (called by CoreGPU orchestrator) --//
      T testSubset(const Samples<T>& samples, ulong startIdx, ulong endIdx);

      //-- Step-by-step training methods (for external orchestration, e.g., CNN) --//
      Tensor1D<T> backpropagate(const Output<T>& output);
      void accumulate();
      void resetAccumulators();

      //-- Gradient access (for multi-GPU merging) --//
      void readAccumulatedGradients(Tensor1D<T>& accumWeights, Tensor1D<T>& accumBiases);
      void setAccumulators(const Tensor1D<T>& accumWeights, const Tensor1D<T>& accumBiases);

      //-- Weight update --//
      void update(ulong numSamples);

      //-- Parameter synchronization --//
      void syncParametersFromGPU();

      //-- Parameter access --//
      const Parameters<T>& getParameters() const { return parameters; }

      //-- Shared-core integration: source loading and buffer allocation --//
      void loadSources(bool skipDefines);
      void allocateBuffers();

      //-- Shared-core integration: kernel building blocks --//
      void addPropagateKernels();
      void addBackpropagateKernels(bool includeInputGradients);
      void addAccumulateKernels();
      void addUpdateKernels(ulong numSamples);

      //-- Shared-core integration: data access --//
      Output<T> readOutput();
      Tensor1D<T> readInputGradients();

    private:
      //-- Configuration --//
      LayersConfig layersConfig;
      TrainingConfig<T> trainingConfig;
      Parameters<T> parameters;
      ulong progressReports = 1000;
      LogLevel logLevel = LogLevel::ERROR;

      //-- OpenCL state --//
      std::unique_ptr<OpenCLWrapper::Core> ownedCore;  // Owned core (standalone mode)
      OpenCLWrapper::Core* core = nullptr;              // Pointer to active core (owned or shared)

      //-- Kernel setup flags --//
      bool predictKernelsSetup = false;
      bool trainingKernelsSetup = false;
      bool backpropagateKernelsSetup = false;
      bool accumulateKernelsSetup = false;
      bool updateKernelsSetup = false;

      //-- Initialization --//
      void initializeParameters();

      //-- Kernel setup (standalone mode) --//
      void setupPredictKernels();
      void setupTrainingKernels();
      void setupBackpropagateKernels();
      void setupAccumulateKernels();
      void setupUpdateKernels(ulong numSamples);

      //-- Helpers --//
      void invalidateAllKernelFlags();

      //-- Loss --//
      T calculateLoss(const Output<T>& expected);
  };
}

#endif // ANN_COREGPUWORKER_H

