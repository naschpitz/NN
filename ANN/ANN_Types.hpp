#ifndef ANN_TYPES_HPP
#define ANN_TYPES_HPP

#include <vector>

//===================================================================================================================//

namespace ANN
{
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

  // Result of a single-sample training step (forward + backward + accumulate).
  // Used by external orchestrators that embed ANN as a sub-network (e.g., a dense
  // head after convolutional layers) and need the predicted output for loss
  // calculation plus the input-layer gradients to continue backpropagation
  // through their own preceding layers.
  template <typename T>
  struct TrainStepResult {
      Output<T> predicted; // Network output from the forward pass
      Tensor1D<T> inputGradients; // dCost/dInput — gradient w.r.t. the input layer
  };
}

//===================================================================================================================//

#endif // ANN_TYPES_HPP
