#ifndef NN_CLI_AUGMENTATIONTRANSFORMS_HPP
#define NN_CLI_AUGMENTATIONTRANSFORMS_HPP

namespace NN_CLI
{

  struct AugmentationTransforms {
      bool horizontalFlip = true; // Mirror along vertical axis (true = enabled)
      float rotation = 15.0f; // Max rotation in degrees (0 = disabled, 15 = ±15°)
      float translation = 0.1f; // Max shift as fraction of image size (0 = disabled, 0.1 = ±10%)
      float brightness = 0.1f; // Max brightness delta (0 = disabled, 0.1 = ±0.1)
      float contrast = 0.2f; // Max contrast delta from 1.0 (0 = disabled, 0.2 = range 0.8–1.2×)
      float gaussianNoise = 0.02f; // Noise standard deviation (0 = disabled, 0.02 = σ=0.02)
  };

} // namespace NN_CLI

#endif // NN_CLI_AUGMENTATIONTRANSFORMS_HPP
