#ifndef ANN_COREGPUWORKER_H
#define ANN_COREGPUWORKER_H

#include "ANN_Worker.hpp"
#include "ANN_Core.hpp"
#include "ANN_GPUBufferManager.hpp"
#include "ANN_GPUKernelBuilder.hpp"

#include <OCLW_Core.hpp>

#include <memory>
#include <utility>

//===================================================================================================================//

namespace ANN
{
  template <typename T>
  class CoreGPUWorker : public Worker<T>
  {
    public:
      // Standalone constructor — creates its own OpenCL core
      CoreGPUWorker(const LayersConfig& layersConfig, const TrainingConfig<T>& trainingConfig,
                    const Parameters<T>& parameters,
                    const CostFunctionConfig<T>& costFunctionConfig = CostFunctionConfig<T>(),
                    ulong progressReports = 1000, LogLevel logLevel = LogLevel::ERROR);

      // Shared-core constructor — uses externally-provided OpenCL core (for CNN integration).
      // Only initializes parameters. Caller must invoke loadSources() and allocateBuffers() manually.
      CoreGPUWorker(const LayersConfig& layersConfig, const TrainingConfig<T>& trainingConfig,
                    const Parameters<T>& parameters, const CostFunctionConfig<T>& costFunctionConfig,
                    OpenCLWrapper::Core& sharedCore, ulong progressReports = 1000, LogLevel logLevel = LogLevel::ERROR);

      //-- Predict (returns post-activation output and pre-activation logits) --//
      PredictResult<T> predict(const Input<T>& input);

      //-- Training (called by CoreGPU orchestrator) --//
      T trainSubset(const Samples<T>& batchSamples, ulong totalSamples, ulong epoch, ulong totalEpochs,
                    const TrainingCallback<T>& callback);

      //-- Testing (called by CoreGPU orchestrator) --//
      std::pair<T, ulong> testSubset(const Samples<T>& samples, ulong startIdx, ulong endIdx);

      //-- Batch predict (called by CoreGPU orchestrator) --//
      PredictResults<T> predictSubset(const Inputs<T>& inputs, ulong startIdx, ulong endIdx,
                                      const ProgressCallback& callback = nullptr);

      //-- Step-by-step training (for external orchestration) --//
      Tensor1D<T> backpropagate(const Output<T>& expected);
      void accumulate();
      void resetAccumulators();

      //-- Weight update --//
      void update(ulong numSamples);

      //-- Parameter access --//
      const Parameters<T>& getParameters() const
      {
        return parameters;
      }

      //-- GPU buffer access (for diagnostics/testing) --//
      template <typename U>
      void readGPUBuffer(const std::string& name, std::vector<U>& hostBuffer, ulong offset = 0)
      {
        this->core->template readBuffer<U>(name, hostBuffer, offset);
      }

      //-- Components (public for direct access by CNN and CoreGPU) --//
      std::unique_ptr<GPUBufferManager<T>> bufferManager;
      std::unique_ptr<GPUKernelBuilder<T>> kernelBuilder;

    private:
      //-- Configuration --//
      LayersConfig layersConfig;
      TrainingConfig<T> trainingConfig;
      Parameters<T> parameters;
      CostFunctionConfig<T> costFunctionConfig;
      ulong progressReports = 1000;
      LogLevel logLevel = LogLevel::ERROR;

      //-- OpenCL state --//
      std::unique_ptr<OpenCLWrapper::Core> ownedCore; // Owned core (standalone mode)
      OpenCLWrapper::Core* core = nullptr; // Pointer to active core (owned or shared)
  };
}

#endif // ANN_COREGPUWORKER_H
