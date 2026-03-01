#ifndef ANN_CORECPU_H
#define ANN_CORECPU_H

#include "ANN_Core.hpp"
#include "ANN_CoreCPUWorker.hpp"

#include <QMutex>

#include <memory>
#include <vector>

//==============================================================================//

namespace ANN
{
  template <typename T>
  class CoreCPU : public Core<T>
  {
    public:
      //-- Constructor --//
      CoreCPU(const CoreConfig<T>& config);

      //-- Core interface --//
      Output<T> predict(const Input<T>& input) override;
      void train(ulong numSamples, const SampleProvider<T>& sampleProvider) override;
      TestResult<T> test(const Samples<T>& samples) override;

      //-- Step-by-step training (for external orchestration, e.g., CNN) --//
      Tensor1D<T> backpropagate(const Output<T>& output) override;
      void accumulate() override;
      void resetAccumulators() override;
      void update(ulong numSamples) override;

    private:
      //-- Step-by-step worker (for predict + step-by-step training path) --//
      std::unique_ptr<CoreCPUWorker<T>> stepWorker;

      //-- Global accumulators (for merging worker results) --//
      Tensor3D<T> accum_dCost_dWeights;
      Tensor2D<T> accum_dCost_dBiases;
      QMutex accumulatorMutex;

      //-- Initialization --//
      void initializeParameters();
      void allocateGlobalAccumulators();

      //-- Training helpers --//
      void resetGlobalAccumulators();
      void mergeWorkerAccumulators(const CoreCPUWorker<T>& worker);
      void reportProgress(ulong currentEpoch, ulong totalEpochs, ulong currentSample, ulong totalSamples, T sampleLoss,
                          T epochLoss, QMutex& callbackMutex);
  };
}

#endif // ANN_CORECPU_H
