#ifndef ANN_COREGPU_H
#define ANN_COREGPU_H

#include "ANN_Core.hpp"

#include <OCLW_Core.hpp>

//===================================================================================================================//

namespace ANN {
  template <typename T>
  class CoreGPU : public Core<T> {
    public:
      CoreGPU(const CoreConfig<T>& config);

      Output<T> run(const Input<T>& input);
      void train(const Samples<T>& samples);

    private:
      OpenCLWrapper::Core oclwCore;

      Tensor1D<T> flatActvs;

      // Kernel setup flags - kernels are set up once and reused
      bool sampleKernelsSetup = false;
      bool updateKernelsSetup = false;

      // Functions used in init()
      void allocateCommon();
      void allocateTraining();

      // Kernel setup functions - called once to create all kernels
      void setupSampleKernels();   // propagate + backpropagate + accumulate kernels
      void setupUpdateKernels(ulong numSamples);  // update kernels (per epoch)

      // Functions used by run()
      void propagate(const Input<T>& input);

      // Functions used by train()
      T calculateLoss(const Output<T>& expected);
      void backpropagate(const Output<T>& output);
      void accumulate();
      void update(ulong numSamples);

      void writeInput(const Input<T>& input);
      Output<T> readOutput();

      void resetAccumulators();
  };
}

#endif // ANN_COREGPU_H
