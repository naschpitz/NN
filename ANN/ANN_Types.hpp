#ifndef ANN_TYPES_HPP
#define ANN_TYPES_HPP

#include <sys/types.h>
#include <vector>

//===================================================================================================================//

namespace ANN
{
  template <typename T>
  using Input = std::vector<T>;

  template <typename T>
  using Inputs = std::vector<Input<T>>;

  template <typename T>
  using Output = std::vector<T>;

  // Pre-activation values (z) of the last layer. Useful for
  // out-of-distribution detection scores (max-logit, logit-norm,
  // free-energy) that cannot be recovered from softmax outputs.
  template <typename T>
  using Logits = std::vector<T>;

  template <typename T>
  using Tensor1D = std::vector<T>;

  template <typename T>
  using Tensor2D = std::vector<std::vector<T>>;

  template <typename T>
  using Tensor3D = std::vector<std::vector<std::vector<T>>>;
}

//===================================================================================================================//

#endif // ANN_TYPES_HPP
