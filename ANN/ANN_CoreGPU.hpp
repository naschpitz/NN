#ifndef ANN_COREGPU_H
#define ANN_COREGPU_H

#include "ANN_Core.hpp"
#include "ANN_CoreGPUWorker.hpp"

#include <memory>
#include <vector>

//===================================================================================================================//

namespace ANN {
  template <typename T>
  class CoreGPU : public Core<T> {
    public:
      CoreGPU(const CoreConfig<T>& config);

      Output<T> predict(const Input<T>& input) override;
      void train(const Samples<T>& samples) override;
      TestResult<T> test(const Samples<T>& samples) override;

      // Step-by-step training methods (for external orchestration, e.g., CNN)
      Tensor1D<T> backpropagate(const Output<T>& output) override;
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
