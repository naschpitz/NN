#ifndef ANN_CORE_H
#define ANN_CORE_H

#include "ANN_ActvFunc.hpp"
#include "ANN_CoreMode.hpp"

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

  struct Layer {
    uint numNeurons;
    ActvFuncType actvFuncType;
  };

  using LayersConfig = std::vector<Layer>;

  template <typename T>
  struct TrainingConfig {
    uint numEpochs;
    float learningRate;
  };

  template <typename T>
  struct Parameters {
    Tensor3D<T> weights;
    Tensor2D<T> biases;
  };

  template <typename T>
  struct CoreConfig {
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
      Core(const CoreConfig<T>& config);
      void init(const CoreConfig<T>& config);

      Output<T> run(const Input<T>& input);
      void train(const Samples<T>& samples);

    private:
      CoreModeType coreModeType;
      LayersConfig layersConfig;
      TrainingConfig<T> trainingConfig;
      Parameters<T> parameters;

      Tensor2D<T> actvs;
      Tensor2D<T> zs;

      Tensor2D<T> dCost_dActvs;
      Tensor3D<T> dCost_dWeights, accum_dCost_dWeights;
      Tensor2D<T> dCost_dBiases, accum_dCost_dBiases;

      // Functions used in init()
      void sanityCheck();
      void allocateCommon();
      void allocateTraining();

      // Functions used by run()
      void propagate(const Input<T>& input);
      Output<T> getOutput();

      // Functions used by train()
      void backpropagate(const Output<T>& output);
      void accumulate();
      void update(uint numSamples);

      // Functions used in backpropagate()
      T calc_dCost_dActv(uint j, const Output<T>& output);
      T calc_dCost_dActv(uint l, uint k);

      T calc_dCost_dWeight(uint l, uint j, uint k);
      T calc_dCost_dBias(uint l, uint j);
  };
}

#endif // ANN_CORE_H
