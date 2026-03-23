#ifndef NN_SERVER_IMAGELOADER_HPP
#define NN_SERVER_IMAGELOADER_HPP

#include <string>
#include <vector>

namespace NN_Server
{

  /**
   * ImageLoader: utility to convert between raw image bytes and flat NCHW float vectors.
   *
   * Supported formats (read): JPEG, PNG, BMP, GIF, TGA, PSD, HDR, PIC
   * Supported formats (write): PNG, JPEG
   *
   * Images are normalised to [0, 1] on load and de-normalised on save.
   * Layout is NCHW: data[c * H * W + h * W + w].
   */
  class ImageLoader
  {
    public:
      // Load an image from raw bytes (e.g. from an HTTP upload) and convert to a flat NCHW float vector.
      // targetC: desired channels (1=grayscale, 3=RGB)
      // targetH, targetW: desired spatial dimensions (resized if necessary)
      static std::vector<float> loadImageFromMemory(const unsigned char* data, int dataSize, int targetC, int targetH,
                                                    int targetW);

      // Encode a flat NCHW float vector ([0,1]) as a PNG image in memory.
      // Returns the raw PNG bytes.
      static std::vector<unsigned char> saveImageToMemory(const std::vector<float>& data, int c, int h, int w);
  };

} // namespace NN_Server

#endif // NN_SERVER_IMAGELOADER_HPP
