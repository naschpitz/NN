#ifndef ANN_CORECPU_H
#define ANN_CORECPU_H

#include "ANN_Core.hpp"
#include "ANN_CoreCPUWorker.hpp"

#include <QMutex>
#include <QThreadPool>

#include <functional>
#include <memory>
#include <vector>

//==============================================================================//

namespace ANN
{
  using namespace Common;
  template <typename T>
  class CoreCPU : public Core<T>
  {
    public:
      //-- Constructor --//
      CoreCPU(const CoreConfig<T>& config);

      //-- Core interface --//
      PredictResults<T> predict(ulong numSamples, const InputProvider<T>& provider) override;
      PredictResult<T> predict(const Input<T>& input) override;
      void train(ulong numSamples, const SampleProvider<T>& sampleProvider) override;
      TestResult<T> test(ulong numSamples, const SampleProvider<T>& sampleProvider) override;

      //-- Step-by-step training (for external orchestration) --//
      Tensor1D<T> backpropagate(const Output<T>& expected) override;
      void accumulate() override;
      void resetAccumulators() override;
      void update(ulong numSamples) override;

    private:
      //-- Persistent worker (for predict and step-by-step training) --//
      std::unique_ptr<CoreCPUWorker<T>> stepWorker;

      //-- Dedicated thread pool for this core's parallel worker dispatch.            --//
      //   Each Core instance owns its own pool instead of sharing the global one, so   //
      //   a validation test() running on a separate core from inside train()'s         //
      //   training callback never contends with train()'s own parallel region. Sharing //
      //   the global pool deadlocks: the nested test() can never acquire a worker       //
      //   thread because they are all occupied by the enclosing train(). Mirrors        //
      //   CNN::CoreCPU.                                                                  //
      QThreadPool workerPool;

      //-- Global accumulators (for merging worker results) --//
      Tensor3D<T> accum_dCost_dWeights;
      Tensor2D<T> accum_dCost_dBiases;
      QMutex accumulatorMutex;

      //-- Adam optimizer state --//
      Tensor3D<T> adam_m_weights; // First moment estimate for weights
      Tensor2D<T> adam_m_biases; // First moment estimate for biases
      Tensor3D<T> adam_v_weights; // Second moment estimate for weights
      Tensor2D<T> adam_v_biases; // Second moment estimate for biases
      ulong adam_t = 0; // Timestep counter

      //-- Initialization --//
      void initializeParameters();
      void allocateGlobalAccumulators();
      void allocateAdamState();

      //-- Run `body(workerIdx)` for workerIdx in [0, numThreads) on this core's        --//
      //   dedicated pool, blocking until all workers finish. Replaces                    //
      //   QtConcurrent::blockingMap, which always uses the (shared) global pool.          //
      //   With numThreads==1 a single worker processes its whole chunk in order, so the   //
      //   bit-deterministic serial behaviour the tests rely on is preserved.              //
      void runWorkers(int numThreads, const std::function<void(int)>& body);

      //-- Training helpers --//
      void resetGlobalAccumulators();
      void mergeWorkerAccumulators(const CoreCPUWorker<T>& worker);
      void reportProgress(ulong currentEpoch, ulong totalEpochs, ulong currentSample, ulong totalSamples, T sampleLoss,
                          T epochLoss, QMutex& callbackMutex);
  };
}

#endif // ANN_CORECPU_H
