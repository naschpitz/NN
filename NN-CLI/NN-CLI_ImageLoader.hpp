#ifndef NN_CLI_IMAGELOADER_HPP
#define NN_CLI_IMAGELOADER_HPP

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
};

} // namespace NN_CLI

//===================================================================================================================//

#endif // NN_CLI_IMAGELOADER_HPP

