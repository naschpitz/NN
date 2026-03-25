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
      using Core<T>::predict; // Bring in the single-input convenience wrapper
      Outputs<T> predict(const Inputs<T>& inputs) override;
      void train(ulong numSamples, const SampleProvider<T>& sampleProvider) override;
      TestResult<T> test(ulong numSamples, const SampleProvider<T>& sampleProvider) override;

      //-- Single-sample training (for external orchestration) --//
      // Performs forward pass, backward pass, and gradient accumulation in one call.
      // Used by external orchestrators that embed ANN as a sub-network (e.g., a dense
      // head after convolutional layers) and need the predicted output for loss
      // calculation plus the input-layer gradients to continue backpropagation
      // through their own preceding layers.
      TrainStepResult<T> trainStep(const Input<T>& input, const Output<T>& expected);
      void resetAccumulators() override;
      void update(ulong numSamples) override;

    private:
      //-- Persistent worker (for predict and single-sample training) --//
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
