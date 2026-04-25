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
      PredictResults<T> predict(const Inputs<T>& inputs) override;
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

      //-- Training helpers --//
      void resetGlobalAccumulators();
      void mergeWorkerAccumulators(const CoreCPUWorker<T>& worker);
      void reportProgress(ulong currentEpoch, ulong totalEpochs, ulong currentSample, ulong totalSamples, T sampleLoss,
                          T epochLoss, QMutex& callbackMutex);
  };
}

#endif // ANN_CORECPU_H
