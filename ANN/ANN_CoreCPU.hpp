#ifndef ANN_CORECPU_H
#define ANN_CORECPU_H

#include "ANN_ActvFunc.hpp"
#include "ANN_Core.hpp"

#include <QMutex>

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
  };

  template <typename T>
  class CoreCPU : public Core<T> {
    public:
      CoreCPU(const CoreConfig<T>& config);

      Output<T> run(const Input<T>& input) override;
      void train(const Samples<T>& samples) override;
      TestResult<T> test(const Samples<T>& samples) override;

    private:
      Tensor2D<T> dCost_dActvs;
      Tensor3D<T> dCost_dWeights, accum_dCost_dWeights;
      Tensor2D<T> dCost_dBiases, accum_dCost_dBiases;

      // Mutex for thread-safe accumulation
      QMutex accumulatorMutex;

      // Functions used in init()
      void allocateCommon();
      void allocateTraining();
      void allocateWorker(SampleWorker<T>& worker);

      // Core computation functions (operate on provided data structures)
      void propagate(const Input<T>& input, Tensor2D<T>& actvs, Tensor2D<T>& zs);
      void backpropagate(const Output<T>& output, const Tensor2D<T>& actvs, const Tensor2D<T>& zs,
                         Tensor2D<T>& dCost_dActvs, Tensor3D<T>& dCost_dWeights, Tensor2D<T>& dCost_dBiases);

      // Functions used in backpropagate() - parameterized versions
      T calc_dCost_dActv(ulong j, const Output<T>& output, const Tensor2D<T>& actvs);
      T calc_dCost_dActv(ulong l, ulong k, const Tensor2D<T>& zs, const Tensor2D<T>& dCost_dActvs);
      T calc_dCost_dWeight(ulong l, ulong j, ulong k, const Tensor2D<T>& actvs, const Tensor2D<T>& zs, const Tensor2D<T>& dCost_dActvs);
      T calc_dCost_dBias(ulong l, ulong j, const Tensor2D<T>& zs, const Tensor2D<T>& dCost_dActvs);

      // Functions used by train()
      void resetAccumulators();
      void resetWorkerAccumulators(SampleWorker<T>& worker);
      void accumulateToWorker(SampleWorker<T>& worker);
      void mergeWorkerAccumulators(const SampleWorker<T>& worker);
      T calculateLoss(const Output<T>& expected, const Tensor2D<T>& actvs);
      void update(ulong numSamples);
      void reportProgress(ulong currentEpoch, ulong totalEpochs, ulong currentSample, ulong totalSamples,
                          T sampleLoss, T epochLoss, QMutex& callbackMutex);

      // Convenience wrappers using member data (for run())
      void propagate(const Input<T>& input);
      Output<T> getOutput();
  };
}

#endif // ANN_CORECPU_H
