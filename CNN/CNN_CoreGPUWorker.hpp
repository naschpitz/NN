#ifndef CNN_COREGPUWORKER_HPP
#define CNN_COREGPUWORKER_HPP

#include "CNN_Core.hpp"
#include "CNN_Worker.hpp"
#include "CNN_CoreGPUWorkerConfig.hpp"
#include "CNN_GPUBufferManager.hpp"
#include "CNN_GPUKernelBuilder.hpp"

#include <ANN_CoreGPUWorker.hpp>
#include <OCLW_Core.hpp>

#include <memory>
#include <span>
#include <utility>

//===================================================================================================================//

namespace CNN
{
  template <typename T>
  class CoreGPUWorker : public Worker<T>
  {
    public:
      // Standalone constructor — creates its own OpenCL core
      CoreGPUWorker(const CoreGPUWorkerConfig<T>& workerConfig);

      // Shared-core constructor — uses externally-provided OpenCL core.
      // Only initializes parameters and computes offsets. Caller must invoke
      // loadSources(), allocateBuffers() manually.
      CoreGPUWorker(const CoreGPUWorkerConfig<T>& workerConfig, OpenCLWrapper::Core& sharedCore);

      //-- Predict (returns post-activation output and dense-head logits) --//
      Common::PredictResult<T> predict(const Input<T>& input);

      //-- Training (called by CoreGPU orchestrator) --//
      // timingCallback/gpuIndex are optional instrumentation: when set, the worker
      // notifies phase boundaries (H2DUpload / GpuCompute) tagged with its GPU index.
      // gpuProfileCallback receives GPU-profiled per-sub-phase kernel times.
      T trainSubset(SamplesView<T> batchSamples, ulong totalSamples, ulong epoch, ulong totalEpochs,
                    const Common::TrainCallback<T>& callback, const TimingCallback& timingCallback = nullptr,
                    int gpuIndex = -1, const GpuProfileCallback& gpuProfileCallback = nullptr,
                    const std::string& gpuProfileDumpPath = "");

      //-- Accumulator reset (called by CoreGPU before trainSubset, mirrors CoreCPUWorker) --//
      void resetAccumulators();
      void resetAccumLoss();

      //-- Testing --//
      std::pair<T, ulong> testSubset(SamplesView<T> samples);

      //-- Batch predict (called by CoreGPU orchestrator) --//
      Common::PredictResults<T> predictSubset(InputsView<T> inputs, const Common::ProgressCallback& callback = nullptr);

      //-- Step-by-step training methods (for external orchestration) --//
      void backpropagateSample(const Input<T>& input, const Output<T>& expected);
      void accumulate();

      //-- Weight update --//
      void update(ulong numSamples);

      //-- Kernel save/restore (delegates to OpenCL core) --//
      std::vector<std::vector<OpenCLWrapper::Kernel>> saveKernels();
      void restoreKernels(const std::vector<std::vector<OpenCLWrapper::Kernel>>& kernels);
      void setTrainKernelsReady(bool ready);

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

      //-- Components (public for direct access by CoreGPU) --//
      std::unique_ptr<GPUBufferManager<T>> bufferManager;
      std::unique_ptr<GPUKernelBuilder<T>> kernelBuilder;

    private:
      //-- Configuration --//
      CoreGPUWorkerConfig<T> workerConfig;
      Parameters<T> parameters;

      //-- OpenCL state --//
      std::unique_ptr<OpenCLWrapper::Core> ownedCore; // Owned core (standalone mode)
      OpenCLWrapper::Core* core = nullptr; // Pointer to active core (owned or shared);

      //-- GPU profiling helper --//
      void collectGpuProfile(const GpuProfileCallback& callback, int gpuIndex, const std::string& gpuProfileDumpPath);
      bool profilingEnabled = false;

      //-- Per-sample progress reporting (loss delta from the GPU accumulator) --//
      void reportSampleProgress(const Common::TrainCallback<T>& callback, ulong currentSample, ulong totalSamples,
                                ulong epoch, ulong totalEpochs, T& prevAccumLoss);
  };
}

//===================================================================================================================//

#endif // CNN_COREGPUWORKER_HPP
