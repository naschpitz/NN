#ifndef ANN_CORECPU_H
#define ANN_CORECPU_H

#include "ANN_ActvFunc.hpp"
#include "ANN_Core.hpp"

//==============================================================================//

namespace ANN {
  template <typename T>
  class CoreCPU : public Core<T> {
    public:
      CoreCPU(const CoreConfig<T>& config);

      Output<T> run(const Input<T>& input);
      void train(const Samples<T>& samples);

    private:
      Tensor2D<T> dCost_dActvs;
      Tensor3D<T> dCost_dWeights, accum_dCost_dWeights;
      Tensor2D<T> dCost_dBiases, accum_dCost_dBiases;

      // Functions used in init()
      void allocateCommon();
      void allocateTraining();

      // Functions used by run()
      void propagate(const Input<T>& input);
      Output<T> getOutput();

      // Functions used by train()
      void backpropagate(const Output<T>& output);
      void accumulate();
      void resetAccumulators();
      void update(ulong numSamples);

      // Functions used in backpropagate()
      T calc_dCost_dActv(ulong j, const Output<T>& output);
      T calc_dCost_dActv(ulong l, ulong k);

      T calc_dCost_dWeight(ulong l, ulong j, ulong k);
      T calc_dCost_dBias(ulong l, ulong j);
  };
}

#endif // ANN_CORECPU_H
