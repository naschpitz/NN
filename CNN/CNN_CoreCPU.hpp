#ifndef CNN_CORECPU_HPP
#define CNN_CORECPU_HPP

#include "CNN_Core.hpp"
#include "CNN_CoreCPUWorker.hpp"

#include <QThreadPool>

#include <functional>
#include <memory>
#include <vector>

//===================================================================================================================//

namespace CNN
{
  using namespace Common;
  template <typename T>
  class CoreCPU : public Core<T>
  {
    public:
      //-- Constructor --//
      CoreCPU(const CoreConfig<T>& config);

      //-- Core interface --//
      using Core<T>::predict; // Bring in the eager Inputs<T> + single-input wrappers
      Common::PredictResults<T> predict(ulong numSamples, const InputProvider<T>& provider) override;
      void train(ulong numSamples, const SampleProvider<T>& sampleProvider) override;
      Common::TestResult<T> test(ulong numSamples, const SampleProvider<T>& sampleProvider) override;

    private:
      //-- Step-by-step worker (for predict / single-threaded path) --//
      std::unique_ptr<CoreCPUWorker<T>> stepWorker;

      //-- Dedicated thread pool for this core's parallel worker dispatch.            --//
      //   Each Core instance owns its own pool instead of sharing the global one, so   //
      //   a validation test() running on a separate core from inside train()'s         //
      //   training callback never contends with train()'s own parallel region. Sharing //
      //   the global pool deadlocks: the nested test() can never acquire a worker       //
      //   thread because they are all occupied by the enclosing train().                //
      QThreadPool workerPool;

      //-- BatchNorm flag (set by scanning layersConfig for BATCHNORM layers) --//
      bool hasBatchNorm = false;

      //-- Global CNN gradient accumulators (for merging worker results) --//
      std::vector<std::vector<T>> accumDConvFilters;
      std::vector<std::vector<T>> accumDConvBiases;
      std::vector<std::vector<T>> accumDBNGamma;
      std::vector<std::vector<T>> accumDBNBeta;
      std::vector<std::vector<T>> accumDResidualWeights;
      std::vector<std::vector<T>> accumDResidualBiases;
      std::vector<std::vector<T>> accumNormMean;
      std::vector<std::vector<T>> accumNormVar;

      //-- Adam optimizer state for CNN conv parameters --//
      std::vector<std::vector<T>> adam_m_filters;
      std::vector<std::vector<T>> adam_v_filters;
      std::vector<std::vector<T>> adam_m_biases;
      std::vector<std::vector<T>> adam_v_biases;
      std::vector<std::vector<T>> adam_m_norm_gamma;
      std::vector<std::vector<T>> adam_v_norm_gamma;
      std::vector<std::vector<T>> adam_m_norm_beta;
      std::vector<std::vector<T>> adam_v_norm_beta;
      std::vector<std::vector<T>> adam_m_residual_weights;
      std::vector<std::vector<T>> adam_v_residual_weights;
      std::vector<std::vector<T>> adam_m_residual_biases;
      std::vector<std::vector<T>> adam_v_residual_biases;
      ulong adam_t = 0;

      //-- Run `body(workerIdx)` for workerIdx in [0, numThreads) on this core's        --//
      //   dedicated pool, blocking until all workers finish. Replaces                    //
      //   QtConcurrent::blockingMap, which always uses the (shared) global pool.          //
      void runWorkers(int numThreads, const std::function<void(int)>& body);

      //-- Training helpers --//
      void resetGlobalCNNAccumulators();
      void mergeWorkerCNNAccumulators(const CoreCPUWorker<T>& worker);
      void updateCNNParameters(ulong numSamples);
      void updateNormRunningStats(ulong numSamples);
      void allocateAdamState();

      //-- BatchNorm-aware training (layer-by-layer orchestration) --//
      void trainBatchNorm(ulong numSamples, const SampleProvider<T>& sampleProvider);
  };
}

//===================================================================================================================//

#endif // CNN_CORECPU_HPP
