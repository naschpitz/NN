#include "NN-Server_ImageLoader.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

#include <algorithm>
#include <stdexcept>

namespace NN_Server
{

  //===================================================================================================================//

  std::vector<float> ImageLoader::loadImageFromMemory(const unsigned char* data, int dataSize,
                                                      int targetC, int targetH, int targetW)
  {
    int origW = 0, origH = 0, origC = 0;
    unsigned char* pixels = stbi_load_from_memory(data, dataSize, &origW, &origH, &origC, targetC);

    if (!pixels) {
      throw std::runtime_error(std::string("Failed to decode image: ") + stbi_failure_reason());
    }

    // Resize if the loaded image doesn't match target dimensions
    std::vector<unsigned char> resizedBuf;
    unsigned char* source = pixels;

    if (origW != targetW || origH != targetH) {
      resizedBuf.resize(static_cast<size_t>(targetW) * targetH * targetC);

      stbir_pixel_layout layout;

      if (targetC == 1)
        layout = STBIR_1CHANNEL;
      else if (targetC == 3)
        layout = STBIR_RGB;
      else if (targetC == 4)
        layout = STBIR_RGBA;
      else
        layout = STBIR_1CHANNEL; // fallback

      stbir_resize_uint8_linear(pixels, origW, origH, 0, resizedBuf.data(), targetW, targetH, 0, layout);
      source = resizedBuf.data();
    }

    // Convert to flat NCHW float vector, normalised to [0, 1]
    std::vector<float> result(static_cast<size_t>(targetC) * targetH * targetW);

    for (int c = 0; c < targetC; ++c) {
      for (int h = 0; h < targetH; ++h) {
        for (int w = 0; w < targetW; ++w) {
          // stb_image stores as interleaved HWC: pixel[h * W * C + w * C + c]
          float val = static_cast<float>(source[h * targetW * targetC + w * targetC + c]) / 255.0f;
          // NCHW layout: data[c * H * W + h * W + w]
          result[c * targetH * targetW + h * targetW + w] = val;
        }
      }
    }

    stbi_image_free(pixels);
    return result;
  }

  //===================================================================================================================//

  // Callback for stbi_write_png_to_func — appends bytes to a vector
  static void writeToVector(void* context, void* data, int size)
  {
    auto* vec = static_cast<std::vector<unsigned char>*>(context);
    auto* bytes = static_cast<unsigned char*>(data);
    vec->insert(vec->end(), bytes, bytes + size);
  }

  //===================================================================================================================//

  std::vector<unsigned char> ImageLoader::saveImageToMemory(const std::vector<float>& data, int c, int h, int w)
  {
    // Convert from NCHW float [0,1] to interleaved HWC uint8 [0,255]
    std::vector<unsigned char> pixels(static_cast<size_t>(c) * h * w);

    for (int ch = 0; ch < c; ++ch) {
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          float val = data[ch * h * w + y * w + x];
          val = std::max(0.0f, std::min(1.0f, val));
          pixels[y * w * c + x * c + ch] = static_cast<unsigned char>(val * 255.0f + 0.5f);
        }
      }
    }

    // Encode as PNG to memory
    std::vector<unsigned char> pngData;
    int stride = w * c;
    int result = stbi_write_png_to_func(writeToVector, &pngData, w, h, c, pixels.data(), stride);

    if (!result) {
      throw std::runtime_error("Failed to encode image to PNG");
    }

    return pngData;
  }

  //===================================================================================================================//

} // namespace NN_Server

