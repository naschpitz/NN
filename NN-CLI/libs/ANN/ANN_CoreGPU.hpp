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

      // Functions used in init()
      void allocateCommon();
      void allocateTraining();

      // Functions used by run()
      void propagate(const Input<T>& input);

      // Functions used by train()
      void backpropagate(const Output<T>& output);
      void accumulate();
      void update(ulong numSamples);

      void writeInput(const Input<T>& input);
      Output<T> readOutput();

      void resetAccumulators();
  };
}

#endif // ANN_COREGPU_H
