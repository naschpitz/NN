#ifndef CNN_TYPES_HPP
#define CNN_TYPES_HPP

#include "CNN_Shape3D.hpp"
#include "CNN_Tensor3D.hpp"

#include <span>
#include <sys/types.h>
#include <vector>

//===================================================================================================================//

namespace CNN
{
  // 1D tensor (used for flattened output)
  template <typename T>
  using Tensor1D = std::vector<T>;

  // Input to the CNN is a 3D tensor
  template <typename T>
  using Input = Tensor3D<T>;

  template <typename T>
  using Inputs = std::vector<Input<T>>;

  // Non-owning view over a sequence of inputs (e.g. a sub-batch handed to a worker).
  template <typename T>
  using InputsView = std::span<const Input<T>>;

  // Output from the CNN is a 1D vector (from dense layers)
  template <typename T>
  using Output = std::vector<T>;

  // Pre-activation values (z) of the  dense head's last layer.
  // Useful for OOD-detection scores (max-logit, logit-norm, free-energy)
  // that softmax discards.
  template <typename T>
  using Logits = std::vector<T>;
}

//===================================================================================================================//

#endif // CNN_TYPES_HPP
