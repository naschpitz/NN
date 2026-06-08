#ifndef CNN_COREGPU_HPP
#define CNN_COREGPU_HPP

#include "CNN_Core.hpp"
#include "CNN_CoreGPUWorker.hpp"

#include <QThreadPool>

#include <memory>
#include <vector>

//===================================================================================================================//

namespace CNN
{
  using namespace Common;
  template <typename T>
  class CoreGPU : public Core<T>
  {
    public:
      CoreGPU(const CoreConfig<T>& config);

      using Core<T>::predict; // Bring in eager Inputs<T> + single-input convenience wrappers
      Common::PredictResults<T> predict(ulong numSamples, const InputProvider<T>& provider) override;
      void train(ulong numSamples, const SampleProvider<T>& sampleProvider) override;
      Common::TestResult<T> test(ulong numSamples, const SampleProvider<T>& sampleProvider) override;
      void syncParametersToGPU() override;

      //-- Worker access (for diagnostics/testing) --//
      CoreGPUWorker<T>* getWorker(size_t idx = 0)
      {
        return (idx < gpuWorkers.size()) ? gpuWorkers[idx].get() : nullptr;
      }

    private:
      //-- GPU workers (one per GPU) --//
      std::vector<std::unique_ptr<CoreGPUWorker<T>>> gpuWorkers;
      size_t numGPUs = 1;

      //-- Dedicated pool for cross-GPU dispatch. Each core owns its own pool (not the global
      //   one) so a validation test() run from inside train()'s callback never starves on
      //   worker threads held by the enclosing train(). Dispatched via QtConcurrent::blockingMap.
      QThreadPool workerPool_;

      //-- Initialization --//
      void initializeWorkers();

      //-- Training coordination --//
      void mergeCNNGradients();
      void mergeGradients();
  };
}

//===================================================================================================================//

#endif // CNN_COREGPU_HPP
