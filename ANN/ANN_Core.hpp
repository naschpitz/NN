#ifndef ANN_CORE_H
#define ANN_CORE_H

#include "ANN_ActvFunc.hpp"
#include "ANN_CoreMode.hpp"
#include "ANN_CoreType.hpp"
#include "ANN_LayersConfig.hpp"

#include <memory>
#include <vector>

//==============================================================================//

namespace ANN {
  template <typename T>
  using Input = std::vector<T>;

  template <typename T>
  using Output = std::vector<T>;

  template <typename T>
  using Inputs = std::vector<Input<T>>;

  template <typename T>
  using Outputs = std::vector<Output<T>>;

  template <typename T>
  using Tensor1D = std::vector<T>;

  template <typename T>
  using Tensor2D = std::vector<std::vector<T>>;

  template <typename T>
  using Tensor3D = std::vector<std::vector<std::vector<T>>>;

  template <typename T>
  struct TrainingConfig {
    ulong numEpochs;
    float learningRate;
  };

  template <typename T>
  struct Parameters {
    Tensor3D<T> weights;
    Tensor2D<T> biases;
  };

  template <typename T>
  struct CoreConfig {
    CoreTypeType coreTypeType;
    CoreModeType coreModeType;
    LayersConfig layersConfig;
    TrainingConfig<T> trainingConfig;
    Parameters<T> parameters;
  };

  template <typename T>
  struct Sample {
    Input<T> input;
    Output<T> output;
  };

  template <typename T>
  using Samples = std::vector<Sample<T>>;

  template <typename T>
  class Core {
    public:
      static std::unique_ptr<Core<T>> makeCore(const CoreConfig<T>& config);

      virtual Output<T> run(const Input<T>& input) = 0;
      virtual void train(const Samples<T>& samples) = 0;

      const LayersConfig& getLayersConfig() const { return layersConfig; }
      const TrainingConfig<T>& getTrainingConfig() const { return trainingConfig; }
      const Parameters<T>& getParameters() const { return parameters; }

    protected:
      explicit Core(const CoreConfig<T>& coreConfig);
      void sanityCheck(const CoreConfig<T>& coreConfig);

      CoreTypeType coreTypeType;
      CoreModeType coreModeType;
      LayersConfig layersConfig;
      TrainingConfig<T> trainingConfig;
      Parameters<T> parameters;

      Tensor2D<T> actvs;
      Tensor2D<T> zs;
  };
}

#endif // ANN_CORE_H
