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

    private:
      //-- GPU workers (one per GPU) --//
      std::vector<std::unique_ptr<CoreGPUWorker<T>>> gpuWorkers;
      size_t numGPUs = 1;

      //-- Initialization --//
      void initializeWorkers();

      //-- Training coordination --//
      void mergeGradients();
      void update(ulong numSamples);
  };
}

#endif // ANN_COREGPU_H
