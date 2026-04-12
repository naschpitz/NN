#ifndef NN_CLI_AUGMENTATIONCONFIG_HPP
#define NN_CLI_AUGMENTATIONCONFIG_HPP

#include "NN-CLI_AugmentationTransforms.hpp"
#include "NN-CLI_ValidationDatasetConfig.hpp"

#include <sys/types.h>

namespace NN_CLI
{

  using ulong = unsigned long;

  struct AugmentationConfig {
      ulong augmentationFactor = 0; // 0 = disabled; N = N× total samples per class
      bool balanceAugmentation = false; // true = augment minority classes up to max class count
      bool autoClassWeights = false; // true = auto-compute inverse-frequency class weights
      float augmentationProbability = 0.5f; // Probability of applying each enabled transform (default 50%)
      AugmentationTransforms transforms; // Which transforms to apply and their intensities
      ValidationDatasetConfig validationConfig; // Validation split config
  };

} // namespace NN_CLI

#endif // NN_CLI_AUGMENTATIONCONFIG_HPP
