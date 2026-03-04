#ifndef ANN_GPUKERNELBUILDER_HPP
#define ANN_GPUKERNELBUILDER_HPP

#include "ANN_Types.hpp"
#include "ANN_LayersConfig.hpp"
#include "ANN_TrainingConfig.hpp"
#include "ANN_Parameters.hpp"
#include "ANN_CostFunctionConfig.hpp"
#include "ANN_GPUBufferManager.hpp"
#include "ANN_LogLevel.hpp"

#include <OCLW_Core.hpp>

//===================================================================================================================//

namespace ANN
{
  // Manages OpenCL kernel creation, argument binding, and setup orchestration.
  // Extracted from CoreGPUWorker to reduce class size.
  template <typename T>
  class GPUKernelBuilder
  {
    public:
      GPUKernelBuilder(OpenCLWrapper::Core* core, const LayersConfig& layersConfig, const Parameters<T>& parameters,
                       const TrainingConfig<T>& trainingConfig, const CostFunctionConfig<T>& costFunctionConfig,
                       GPUBufferManager<T>& bufferManager, LogLevel logLevel);

      //-- Kernel setup (clears previous kernels and rebuilds) --//
      void setupPredictKernels();
      void setupTrainingKernels();
      void setupBackpropagateKernels();
      void setupAccumulateKernels();
      void setupUpdateKernels(ulong numSamples);

      //-- Kernel building blocks (public for CNN integration) --//
      void addPropagateKernels(bool training = false);
      void addBackpropagateKernels(bool includeInputGradients);
      void addAccumulateKernels();
      void addUpdateKernels(ulong numSamples);

      //-- Kernel setup flags --//
      bool predictKernelsSetup = false;
      bool trainingKernelsSetup = false;
      bool backpropagateKernelsSetup = false;
      bool accumulateKernelsSetup = false;
      bool updateKernelsSetup = false;

      void invalidateAllKernelFlags();

    private:
      OpenCLWrapper::Core* core;
      const LayersConfig& layersConfig;
      const Parameters<T>& parameters;
      const TrainingConfig<T>& trainingConfig;
      const CostFunctionConfig<T>& costFunctionConfig;
      GPUBufferManager<T>& bufferManager;
      LogLevel logLevel;

      //-- Adam optimizer state --//
      ulong adam_t = 0;
  };
}

//===================================================================================================================//

#endif // ANN_GPUKERNELBUILDER_HPP
