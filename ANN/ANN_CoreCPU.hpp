#ifndef ANN_CORECPU_H
#define ANN_CORECPU_H

#include "ANN_ActvFunc.hpp"
#include "ANN_Core.hpp"

#include <QMutex>

#include <random>

//==============================================================================//

namespace ANN {
  // Worker struct that holds thread-local data for processing samples
  template <typename T>
  struct SampleWorker {
    Tensor2D<T> actvs;
    Tensor2D<T> zs;
    Tensor2D<T> dCost_dActvs;
    Tensor3D<T> dCost_dWeights;
    Tensor2D<T> dCost_dBiases;
    T sampleLoss;
    // Thread-local accumulators to reduce mutex contention
    Tensor3D<T> accum_dCost_dWeights;
    Tensor2D<T> accum_dCost_dBiases;
    T accum_loss;  // Thread-local loss accumulator
    // Dropout masks: dropoutMasks[l][j] = 0 (dropped) or 1/(1-p) (kept, with inverted scaling)
    Tensor2D<T> dropoutMasks;
    std::mt19937 rng{std::random_device{}()};
  };

  template <typename T>
  class CoreCPU : public Core<T> {
    public:
      //-- Constructor --//
      CoreCPU(const CoreConfig<T>& config);

      //-- Core interface --//
      Output<T> predict(const Input<T>& input) override;
      void train(const Samples<T>& samples) override;
      TestResult<T> test(const Samples<T>& samples) override;

      //-- Step-by-step training (for external orchestration, e.g., CNN) --//
      Tensor1D<T> backpropagate(const Output<T>& output) override;
      void accumulate() override;
      void resetAccumulators() override;
      void update(ulong numSamples) override;

    private:
      //-- Gradient state --//
      Tensor2D<T> dCost_dActvs;
      Tensor3D<T> dCost_dWeights, accum_dCost_dWeights;
      Tensor2D<T> dCost_dBiases, accum_dCost_dBiases;
      QMutex accumulatorMutex;

      //-- Dropout state (for step-by-step path used by CNN) --//
      Tensor2D<T> dropoutMasks;
      std::mt19937 dropoutRng{std::random_device{}()};

      //-- Allocation --//
      void allocateCommon();
      void allocateTraining();
      void allocateWorker(SampleWorker<T>& worker);

      //-- Forward / Backward pass --//
      void propagate(const Input<T>& input, Tensor2D<T>& actvs, Tensor2D<T>& zs,
                     bool applyDropout = false, Tensor2D<T>* dropoutMasks = nullptr, std::mt19937* rng = nullptr);
      void backpropagate(const Output<T>& output, const Tensor2D<T>& actvs, const Tensor2D<T>& zs,
                         Tensor2D<T>& dCost_dActvs, Tensor3D<T>& dCost_dWeights, Tensor2D<T>& dCost_dBiases,
                         const Tensor2D<T>* dropoutMasks = nullptr);

      //-- Gradient helpers --//
      T calc_dCost_dActv(ulong j, const Output<T>& output, const Tensor2D<T>& actvs);
      T calc_dCost_dActv(ulong l, ulong k, const Tensor2D<T>& actvs, const Tensor2D<T>& zs, const Tensor2D<T>& dCost_dActvs);
      T calc_dCost_dWeight(ulong l, ulong j, ulong k, const Tensor2D<T>& actvs, const Tensor2D<T>& zs, const Tensor2D<T>& dCost_dActvs);
      T calc_dCost_dBias(ulong l, ulong j, const Tensor2D<T>& actvs, const Tensor2D<T>& zs, const Tensor2D<T>& dCost_dActvs);

      //-- Training helpers --//
      void resetWorkerAccumulators(SampleWorker<T>& worker);
      void accumulateToWorker(SampleWorker<T>& worker);
      void mergeWorkerAccumulators(const SampleWorker<T>& worker);
      T calculateLoss(const Output<T>& expected, const Tensor2D<T>& actvs);
      void reportProgress(ulong currentEpoch, ulong totalEpochs, ulong currentSample, ulong totalSamples,
                          T sampleLoss, T epochLoss, QMutex& callbackMutex);

      //-- Convenience wrappers (for predict) --//
      void propagate(const Input<T>& input);
      Output<T> getOutput();
  };
}

#endif // ANN_CORECPU_H
