#ifndef ANN_COREGPUWORKER_H
#define ANN_COREGPUWORKER_H

#include "ANN_Core.hpp"

#include <OCLW_Core.hpp>

//===================================================================================================================//

namespace ANN {
  template <typename T>
  class CoreGPUWorker {
    public:
      CoreGPUWorker(const LayersConfig& layersConfig, const TrainingConfig<T>& trainingConfig,
                    const Parameters<T>& parameters, bool verbose = false);

      //-- Predict --//
      Output<T> predict(const Input<T>& input);

      //-- Training (called by CoreGPU orchestrator) --//
      T trainSubset(const Samples<T>& samples, ulong startIdx, ulong endIdx, ulong epoch, ulong totalEpochs,
                    const TrainingCallback<T>& callback);

      //-- Testing (called by CoreGPU orchestrator) --//
      T testSubset(const Samples<T>& samples, ulong startIdx, ulong endIdx);

      //-- Gradient access (for multi-GPU merging) --//
      void readAccumulatedGradients(Tensor1D<T>& accumWeights, Tensor1D<T>& accumBiases);
      void setAccumulators(const Tensor1D<T>& accumWeights, const Tensor1D<T>& accumBiases);

      //-- Weight update --//
      void update(ulong numSamples);

      //-- Parameter synchronization --//
      void syncParametersFromGPU();

      //-- Parameter access --//
      const Parameters<T>& getParameters() const { return parameters; }

    private:
      //-- Configuration --//
      LayersConfig layersConfig;
      TrainingConfig<T> trainingConfig;
      Parameters<T> parameters;
      bool verbose = false;

      //-- OpenCL state --//
      OpenCLWrapper::Core oclwCore;

      //-- Kernel setup flags --//
      bool predictKernelsSetup = false;
      bool trainingKernelsSetup = false;
      bool updateKernelsSetup = false;

      //-- Initialization --//
      void allocateCommon();
      void allocateTraining();

      //-- Kernel setup --//
      void setupPredictKernels();
      void setupTrainingKernels();
      void setupUpdateKernels(ulong numSamples);

      //-- Loss and output --//
      T calculateLoss(const Output<T>& expected);
      void resetAccumulators();
      Output<T> readOutput();
  };
}

#endif // ANN_COREGPUWORKER_H

