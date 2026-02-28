#ifndef ANN_COREGPUWORKER_H
#define ANN_COREGPUWORKER_H

#include "ANN_Core.hpp"

#include <OCLW_Core.hpp>

#include <memory>
#include <random>
#include <utility>

//===================================================================================================================//

namespace ANN {
  template <typename T>
  class CoreGPUWorker {
    public:
      // Standalone constructor — creates its own OpenCL core
      CoreGPUWorker(const LayersConfig& layersConfig, const TrainingConfig<T>& trainingConfig,
                    const Parameters<T>& parameters, const CostFunctionConfig<T>& costFunctionConfig = CostFunctionConfig<T>(),
                    ulong progressReports = 1000,
                    LogLevel logLevel = LogLevel::ERROR);

      // Shared-core constructor — uses externally-provided OpenCL core (for CNN integration).
      // Only initializes parameters. Caller must invoke loadSources() and allocateBuffers() manually.
      CoreGPUWorker(const LayersConfig& layersConfig, const TrainingConfig<T>& trainingConfig,
                    const Parameters<T>& parameters, const CostFunctionConfig<T>& costFunctionConfig,
                    OpenCLWrapper::Core& sharedCore,
                    ulong progressReports = 1000,
                    LogLevel logLevel = LogLevel::ERROR);

      //-- Predict --//
      Output<T> predict(const Input<T>& input);

      //-- Training (called by CoreGPU orchestrator) --//
      T trainSubset(const Samples<T>& samples, const std::vector<ulong>& indices,
                    ulong startIdx, ulong endIdx, ulong epoch, ulong totalEpochs,
                    const TrainingCallback<T>& callback);

      //-- Testing (called by CoreGPU orchestrator) --//
      std::pair<T, ulong> testSubset(const Samples<T>& samples, ulong startIdx, ulong endIdx);

      //-- Step-by-step training methods (for external orchestration, e.g., CNN) --//
      Tensor1D<T> backpropagate(const Output<T>& output);
      void accumulate();
      void resetAccumulators();

      //-- Gradient access (for multi-GPU merging) --//
      void readAccumulatedGradients(Tensor1D<T>& accumWeights, Tensor1D<T>& accumBiases);
      void setAccumulators(const Tensor1D<T>& accumWeights, const Tensor1D<T>& accumBiases);

      //-- Dropout --//
      bool hasDropout = false;
      void generateAndUploadDropoutMask();

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
      CostFunctionConfig<T> costFunctionConfig;
      ulong progressReports = 1000;
      LogLevel logLevel = LogLevel::ERROR;

      //-- OpenCL state --//
      std::unique_ptr<OpenCLWrapper::Core> ownedCore;  // Owned core (standalone mode)
      OpenCLWrapper::Core* core = nullptr;              // Pointer to active core (owned or shared)

      //-- Batch parameters --//
      ulong stride = 0;           // totalNumNeurons — offset between samples in batched buffers
      ulong currentBatchSize = 1; // Current batch size for kernel dispatch

    public:
      ulong getStride() const { return this->stride; }
      void setCurrentBatchSize(ulong bs) { this->currentBatchSize = bs; this->invalidateAllKernelFlags(); }
      Output<T> readBatchOutput(ulong batchIdx);
    private:

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
      T calculateBatchLoss(ulong batchSize);

      std::mt19937 dropoutRng{std::random_device{}()};
  };
}

#endif // ANN_COREGPUWORKER_H

