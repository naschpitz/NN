#ifndef ANN_CORECPUWORKER_H
#define ANN_CORECPUWORKER_H

#include "ANN_Worker.hpp"
#include "ANN_ActvFunc.hpp"
#include "ANN_Types.hpp"
#include "ANN_LayersConfig.hpp"
#include "ANN_TrainingConfig.hpp"
#include "ANN_Parameters.hpp"

#include <random>

//===================================================================================================================//

namespace ANN {
  template <typename T>
  class CoreCPUWorker : public Worker<T> {
    public:
      CoreCPUWorker(const LayersConfig& layersConfig,
                    const TrainingConfig<T>& trainingConfig,
                    const Parameters<T>& parameters,
                    const CostFunctionConfig<T>& costFunctionConfig,
                    bool allocateTrainingBuffers = true);

      //-- Forward pass --//
      void propagate(const Input<T>& input, bool applyDropout = false);

      //-- Backward pass --//
      void backpropagate(const Output<T>& output);

      //-- Backward pass returning input gradients (for CNN step-by-step) --//
      Tensor1D<T> backpropagateAndReturnInputGradients(const Output<T>& output);

      //-- Loss --//
      T computeLoss(const Output<T>& expected);

      //-- Accumulation --//
      void accumulate();
      void resetAccumulators();

      //-- Output access --//
      Output<T> getOutput() const;

      //-- Activation access (for test accuracy check) --//
      const Tensor2D<T>& getActvs() const { return actvs; }

      //-- Accumulator access (for merging by CoreCPU) --//
      const Tensor3D<T>& getAccumWeights() const { return accum_dCost_dWeights; }
      const Tensor2D<T>& getAccumBiases() const { return accum_dCost_dBiases; }
      T getAccumLoss() const { return accum_loss; }
      void resetAccumLoss() { accum_loss = 0; }
      void addToAccumLoss(T loss) { accum_loss += loss; }

    private:
      //-- Shared references (owned by CoreCPU/Core) --//
      const LayersConfig& layersConfig;
      const TrainingConfig<T>& trainingConfig;
      const Parameters<T>& parameters;

      //-- Per-worker state --//
      Tensor2D<T> actvs;
      Tensor2D<T> zs;
      Tensor2D<T> dCost_dActvs;
      Tensor3D<T> dCost_dWeights;
      Tensor2D<T> dCost_dBiases;

      //-- Per-worker accumulators --//
      Tensor3D<T> accum_dCost_dWeights;
      Tensor2D<T> accum_dCost_dBiases;
      T accum_loss = 0;

      //-- Dropout --//
      Tensor2D<T> dropoutMasks;
      std::mt19937 rng{std::random_device{}()};

      //-- Gradient helpers --//
      T calc_dCost_dActv(ulong j, const Output<T>& output);
      T calc_dCost_dActv(ulong l, ulong k);
      T calc_dCost_dWeight(ulong l, ulong j, ulong k);
      T calc_dCost_dBias(ulong l, ulong j);

      //-- Allocation --//
      void allocate(bool allocateTrainingBuffers);
  };
}

#endif // ANN_CORECPUWORKER_H

