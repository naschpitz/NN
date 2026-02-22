#include "NN-CLI_ImageLoader.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

#include <QDir>
#include <QFileInfo>

#include <algorithm>
#include <stdexcept>

namespace NN_CLI {

//===================================================================================================================//

std::vector<float> ImageLoader::loadImage(const std::string& imagePath,
                                           int targetC, int targetH, int targetW) {
  int origW = 0, origH = 0, origC = 0;
  unsigned char* pixels = stbi_load(imagePath.c_str(), &origW, &origH, &origC, targetC);

  if (!pixels) {
    throw std::runtime_error("Failed to load image: " + imagePath +
                              " (" + stbi_failure_reason() + ")");
  }

  // Resize if the loaded image doesn't match target dimensions
  std::vector<unsigned char> resizedBuf;
  unsigned char* source = pixels;

  if (origW != targetW || origH != targetH) {
    resizedBuf.resize(static_cast<size_t>(targetW) * targetH * targetC);

    stbir_pixel_layout layout;
    if (targetC == 1)      layout = STBIR_1CHANNEL;
    else if (targetC == 3) layout = STBIR_RGB;
    else if (targetC == 4) layout = STBIR_RGBA;
    else                   layout = STBIR_1CHANNEL; // fallback

    stbir_resize_uint8_linear(pixels, origW, origH, 0,
                               resizedBuf.data(), targetW, targetH, 0,
                               layout);
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

void ImageLoader::saveImage(const std::string& imagePath,
                             const std::vector<float>& data,
                             int c, int h, int w) {
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

  // Determine format from extension
  std::string ext;
  auto dot = imagePath.find_last_of('.');
  if (dot != std::string::npos) {
    ext = imagePath.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  }

  int result = 0;
  if (ext == ".jpg" || ext == ".jpeg") {
    result = stbi_write_jpg(imagePath.c_str(), w, h, c, pixels.data(), 90);
  } else if (ext == ".bmp") {
    result = stbi_write_bmp(imagePath.c_str(), w, h, c, pixels.data());
  } else {
    // Default to PNG
    result = stbi_write_png(imagePath.c_str(), w, h, c, pixels.data(), w * c);
  }

  if (!result) {
    throw std::runtime_error("Failed to save image: " + imagePath);
  }
}

//===================================================================================================================//

std::string ImageLoader::resolvePath(const std::string& imagePath,
                                      const std::string& baseDirPath) {
  QFileInfo fileInfo(QString::fromStdString(imagePath));
  if (fileInfo.isAbsolute()) {
    return imagePath;
  }
  QDir baseDir(QString::fromStdString(baseDirPath));
  return baseDir.filePath(QString::fromStdString(imagePath)).toStdString();
}

//===================================================================================================================//

} // namespace NN_CLI

