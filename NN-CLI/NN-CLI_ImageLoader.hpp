#ifndef NN_CLI_IMAGELOADER_HPP
#define NN_CLI_IMAGELOADER_HPP

#include "NN-CLI_Loader.hpp"

#include <random>
#include <string>
#include <vector>

//===================================================================================================================//

namespace NN_CLI {


/**
 * ImageLoader: utility to load images into flat NCHW float vectors and save vectors as images.
 *
 * Supported formats (read): JPEG, PNG, BMP, GIF, TGA, PSD, HDR, PIC
 * Supported formats (write): PNG, JPEG, BMP
 *
 * Images are normalised to [0, 1] on load and de-normalised on save.
 * Layout is NCHW: data[c * H * W + h * W + w].
 */
class ImageLoader {
public:
  // Load an image and convert to a flat NCHW float vector normalised to [0,1].
  // targetC: desired channels (1=grayscale, 3=RGB)
  // targetH, targetW: desired spatial dimensions (resized if necessary)
  static std::vector<float> loadImage(const std::string& imagePath,
                                       int targetC, int targetH, int targetW);

  // Save a flat NCHW float vector ([0,1]) as an image file.
  // Format determined by extension: .png, .jpg/.jpeg, .bmp (default: PNG).
  static void saveImage(const std::string& imagePath,
                        const std::vector<float>& data,
                        int c, int h, int w);

  // Resolve imagePath relative to baseDirPath (directory).
  // Returns imagePath unchanged if it is already absolute.
  static std::string resolvePath(const std::string& imagePath,
                                  const std::string& baseDirPath);

  //-- Data augmentation transforms (operate on NCHW [0,1] data in-place) --//

  // Apply a random combination of transforms to an NCHW buffer.
  // rng: random engine for reproducibility.
  static void applyRandomTransforms(std::vector<float>& data, int c, int h, int w,
                                     std::mt19937& rng,
                                     const Loader::AugmentationTransforms& transforms = {},
                                     float probability = 0.5f);

  // Individual transforms (all operate on NCHW [0,1] data)
  static void horizontalFlip(std::vector<float>& data, int c, int h, int w);
  static void randomRotation(std::vector<float>& data, int c, int h, int w, float maxDegrees,
                              std::mt19937& rng);
  static void randomBrightness(std::vector<float>& data, int c, int h, int w, float maxDelta,
                                std::mt19937& rng);
  static void randomContrast(std::vector<float>& data, int c, int h, int w,
                              float minFactor, float maxFactor, std::mt19937& rng);
  static void randomTranslation(std::vector<float>& data, int c, int h, int w,
                                 float maxFraction, std::mt19937& rng);
  static void addGaussianNoise(std::vector<float>& data, float stddev, std::mt19937& rng);
};

} // namespace NN_CLI

//===================================================================================================================//

#endif // NN_CLI_IMAGELOADER_HPP

