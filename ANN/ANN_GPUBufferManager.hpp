#ifndef ANN_GPUBUFFERMANAGER_HPP
#define ANN_GPUBUFFERMANAGER_HPP

#include "ANN_Types.hpp"
#include "ANN_LayersConfig.hpp"
#include "ANN_TrainingConfig.hpp"
#include "ANN_Parameters.hpp"
#include "ANN_CostFunctionConfig.hpp"
#include "ANN_LogLevel.hpp"

#include <OCLW_Core.hpp>

#include <random>

//===================================================================================================================//

namespace ANN
{
  // Manages GPU buffer allocation, source loading, parameter initialization/synchronization,
  // data I/O (read output, read gradients), offset computation, and dropout mask generation.
  // Extracted from CoreGPUWorker to reduce class size.
  template <typename T>
  class GPUBufferManager
  {
    public:
      GPUBufferManager(OpenCLWrapper::Core* core, const LayersConfig& layersConfig, Parameters<T>& parameters,
                       const TrainingConfig<T>& trainingConfig, const CostFunctionConfig<T>& costFunctionConfig,
                       LogLevel logLevel);

      //-- Initialization --//
      void initializeParameters();
      void loadSources(bool skipDefines);
      void allocateBuffers();

      //-- Parameter synchronization --//
      void syncParametersFromGPU(); // GPU → CPU
      void syncParametersToGPU(); // CPU → GPU

      //-- Data I/O --//
      Output<T> readOutput();
      // Read pre-activation z-vector of the last layer (input to the final activation function).
      Logits<T> readOutputLogits();
      Tensor1D<T> readInputGradients();

      //-- Gradient access (for multi-GPU merging) --//
      void readAccumulatedGradients(Tensor1D<T>& accumWeights, Tensor1D<T>& accumBiases);
      void setAccumulators(const Tensor1D<T>& accumWeights, const Tensor1D<T>& accumBiases);

      //-- Dropout --//
      bool hasDropout = false;
      void generateAndUploadDropoutMask();

      //-- Offset queries (used by CNN integration) --//
      ulong getOutputActvOffset() const;
      ulong getNumOutputNeurons() const;

      //-- Precomputed offset helpers (used by GPUKernelBuilder) --//
      ulong getActvOffset(ulong layerIdx) const;
      ulong getWeightOffset(ulong layerIdx) const;
      ulong getBiasOffset(ulong layerIdx) const;

    private:
      OpenCLWrapper::Core* core;
      const LayersConfig& layersConfig;
      Parameters<T>& parameters;
      const TrainingConfig<T>& trainingConfig;
      const CostFunctionConfig<T>& costFunctionConfig;
      LogLevel logLevel;

      std::mt19937 dropoutRng{std::random_device{}()};
  };
}

//===================================================================================================================//

#endif // ANN_GPUBUFFERMANAGER_HPP
