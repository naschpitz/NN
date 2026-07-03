#ifndef ANN_COREGPUWORKER_H
#define ANN_COREGPUWORKER_H

#include "ANN_Worker.hpp"
#include "ANN_Core.hpp"
#include "ANN_GPUBufferManager.hpp"
#include "ANN_GPUKernelBuilder.hpp"

#include "Common/Common_TestSubsetResult.hpp"

#include <OCLW_Core.hpp>

#include <memory>
#include <span>
#include <utility>

//===================================================================================================================//

namespace ANN
{
  template <typename T>
  class CoreGPUWorker : public Worker<T>
  {
    public:
      // Standalone constructor — creates its own OpenCL core
      CoreGPUWorker(const LayersConfig& layersConfig, const Common::TrainConfig<T>& trainConfig,
                    const Parameters<T>& parameters,
                    const Common::CostFunctionConfig<T>& costFunctionConfig = Common::CostFunctionConfig<T>(),
                    ulong progressReports = 1000, Common::LogLevel logLevel = Common::LogLevel::ERROR);

      // Shared-core constructor — uses externally-provided OpenCL core (for CNN integration).
      // Only initializes parameters. Caller must invoke loadSources() and allocateBuffers() manually.
      CoreGPUWorker(const LayersConfig& layersConfig, const Common::TrainConfig<T>& trainConfig,
                    const Parameters<T>& parameters, const Common::CostFunctionConfig<T>& costFunctionConfig,
                    OpenCLWrapper::Core& sharedCore, ulong progressReports = 1000,
                    Common::LogLevel logLevel = Common::LogLevel::ERROR);

      //-- Predict (returns post-activation output and pre-activation logits) --//
      Common::PredictResult<T> predict(const Input<T>& input);

      //-- Training (called by CoreGPU orchestrator) --//
      T trainSubset(SamplesView<T> batchSamples, ulong totalSamples, ulong epoch, ulong totalEpochs,
                    const Common::TrainCallback<T>& callback);

      //-- Testing (called by CoreGPU orchestrator) --//
      Common::TestSubsetResult<T> testSubset(SamplesView<T> samples);

      //-- Batch predict (called by CoreGPU orchestrator) --//
      Common::PredictResults<T> predictSubset(InputsView<T> inputs, const Common::ProgressCallback& callback = nullptr);

      //-- Step-by-step training (for external orchestration) --//
      Tensor1D<T> backpropagate(const Output<T>& expected);
      void accumulate();
      void resetAccumulators();

      //-- Weight update --//
      void update(ulong numSamples);

      //-- LR scheduling: update the worker's trainConfig copy and force the update kernels
      //   to rebuild so the new LR is re-baked into their arguments next update().
      void setLearningRate(T lr)
      {
        this->trainConfig.learningRate = lr;
        this->kernelBuilder->updateKernelsSetup = false;
      }

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
      Common::TrainConfig<T> trainConfig;
      Parameters<T> parameters;
      Common::CostFunctionConfig<T> costFunctionConfig;
      ulong progressReports = 1000;
      Common::LogLevel logLevel = Common::LogLevel::ERROR;

      //-- OpenCL state --//
      std::unique_ptr<OpenCLWrapper::Core> ownedCore; // Owned core (standalone mode)
      OpenCLWrapper::Core* core = nullptr; // Pointer to active core (owned or shared)
  };
}

#endif // ANN_COREGPUWORKER_H
