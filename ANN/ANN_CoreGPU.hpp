#ifndef ANN_COREGPU_H
#define ANN_COREGPU_H

#include "ANN_Core.hpp"
#include "ANN_CoreGPUWorker.hpp"

#include <memory>
#include <vector>

//===================================================================================================================//

namespace ANN
{
  template <typename T>
  class CoreGPU : public Core<T>
  {
    public:
      CoreGPU(const CoreConfig<T>& config);

      using Core<T>::predict; // Bring in the single-input convenience wrapper
      PredictResults<T> predict(const Inputs<T>& inputs) override;
      void train(ulong numSamples, const SampleProvider<T>& sampleProvider) override;
      TestResult<T> test(ulong numSamples, const SampleProvider<T>& sampleProvider) override;

      //-- Step-by-step training (for external orchestration) --//
      Tensor1D<T> backpropagate(const Output<T>& expected) override;
      void accumulate() override;
      void resetAccumulators() override;
      void update(ulong numSamples) override;

    private:
      //-- GPU workers (one per GPU) --//
      std::vector<std::unique_ptr<CoreGPUWorker<T>>> gpuWorkers;
      size_t numGPUs = 1;

      //-- Initialization --//
      void initializeWorkers();

      //-- Training coordination --//
      void mergeGradients();
  };
}

#endif // ANN_COREGPU_H
