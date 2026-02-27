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
#include <cmath>
#include <numeric>
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
//-- Data augmentation transforms --//
//===================================================================================================================//

void ImageLoader::horizontalFlip(std::vector<float>& data, int c, int h, int w) {
  for (int ch = 0; ch < c; ch++) {
    for (int y = 0; y < h; y++) {
      int rowStart = ch * h * w + y * w;
      for (int x = 0; x < w / 2; x++) {
        std::swap(data[rowStart + x], data[rowStart + w - 1 - x]);
      }
    }
  }
}

//===================================================================================================================//

void ImageLoader::randomRotation(std::vector<float>& data, int c, int h, int w,
                                  float maxDegrees, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-maxDegrees, maxDegrees);
  float angle = dist(rng) * static_cast<float>(M_PI) / 180.0f;
  float cosA = std::cos(angle);
  float sinA = std::sin(angle);
  float cx = static_cast<float>(w) / 2.0f;
  float cy = static_cast<float>(h) / 2.0f;

  std::vector<float> result(data.size(), 0.0f);

  for (int ch = 0; ch < c; ch++) {
    int chOffset = ch * h * w;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        // Map destination (x,y) back to source
        float srcX = cosA * (x - cx) + sinA * (y - cy) + cx;
        float srcY = -sinA * (x - cx) + cosA * (y - cy) + cy;

        // Bilinear interpolation
        int x0 = static_cast<int>(std::floor(srcX));
        int y0 = static_cast<int>(std::floor(srcY));
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        float fx = srcX - x0;
        float fy = srcY - y0;

        auto sample = [&](int sx, int sy) -> float {
          if (sx < 0 || sx >= w || sy < 0 || sy >= h) return 0.0f;
          return data[chOffset + sy * w + sx];
        };

        result[chOffset + y * w + x] =
            (1 - fx) * (1 - fy) * sample(x0, y0) +
            fx * (1 - fy) * sample(x1, y0) +
            (1 - fx) * fy * sample(x0, y1) +
            fx * fy * sample(x1, y1);
      }
    }
  }
  data = std::move(result);
}

//===================================================================================================================//

void ImageLoader::randomBrightness(std::vector<float>& data, int /*c*/, int /*h*/, int /*w*/,
                                    float maxDelta, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-maxDelta, maxDelta);
  float delta = dist(rng);
  for (auto& v : data) {
    v = std::clamp(v + delta, 0.0f, 1.0f);
  }
}

//===================================================================================================================//

void ImageLoader::randomContrast(std::vector<float>& data, int c, int h, int w,
                                  float minFactor, float maxFactor, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(minFactor, maxFactor);
  float factor = dist(rng);

  // Compute per-channel mean
  for (int ch = 0; ch < c; ch++) {
    int chOffset = ch * h * w;
    float mean = 0.0f;
    for (int i = 0; i < h * w; i++) mean += data[chOffset + i];
    mean /= static_cast<float>(h * w);

    for (int i = 0; i < h * w; i++) {
      data[chOffset + i] = std::clamp(mean + factor * (data[chOffset + i] - mean), 0.0f, 1.0f);
    }
  }
}

//===================================================================================================================//

void ImageLoader::randomTranslation(std::vector<float>& data, int c, int h, int w,
                                     float maxFraction, std::mt19937& rng) {
  int maxDx = static_cast<int>(maxFraction * w);
  int maxDy = static_cast<int>(maxFraction * h);
  if (maxDx == 0 && maxDy == 0) return;

  std::uniform_int_distribution<int> distX(-maxDx, maxDx);
  std::uniform_int_distribution<int> distY(-maxDy, maxDy);
  int dx = distX(rng);
  int dy = distY(rng);

  std::vector<float> result(data.size(), 0.0f);
  for (int ch = 0; ch < c; ch++) {
    int chOffset = ch * h * w;
    for (int y = 0; y < h; y++) {
      int srcY = y - dy;
      if (srcY < 0 || srcY >= h) continue;
      for (int x = 0; x < w; x++) {
        int srcX = x - dx;
        if (srcX < 0 || srcX >= w) continue;
        result[chOffset + y * w + x] = data[chOffset + srcY * w + srcX];
      }
    }
  }
  data = std::move(result);
}

//===================================================================================================================//

void ImageLoader::addGaussianNoise(std::vector<float>& data, float stddev, std::mt19937& rng) {
  std::normal_distribution<float> dist(0.0f, stddev);
  for (auto& v : data) {
    v = std::clamp(v + dist(rng), 0.0f, 1.0f);
  }
}

//===================================================================================================================//

void ImageLoader::applyRandomTransforms(std::vector<float>& data, int c, int h, int w,
                                         std::mt19937& rng,
                                         const Loader::AugmentationTransforms& transforms) {
  std::bernoulli_distribution coin(0.5);

  // Horizontal flip (applied with probability = horizontalFlip value, 0 = disabled)
  if (transforms.horizontalFlip > 0.0f) {
    std::bernoulli_distribution flipCoin(transforms.horizontalFlip);
    if (flipCoin(rng)) horizontalFlip(data, c, h, w);
  }

  // Random rotation (max degrees from config, 0 = disabled)
  if (transforms.rotation > 0.0f && coin(rng))
    randomRotation(data, c, h, w, transforms.rotation, rng);

  // Random translation (max fraction from config, 0 = disabled)
  if (transforms.translation > 0.0f && coin(rng))
    randomTranslation(data, c, h, w, transforms.translation, rng);

  // Random brightness (max delta from config, 0 = disabled)
  if (transforms.brightness > 0.0f && coin(rng))
    randomBrightness(data, c, h, w, transforms.brightness, rng);

  // Random contrast (delta from 1.0 from config, 0 = disabled)
  if (transforms.contrast > 0.0f && coin(rng)) {
    float minFactor = 1.0f - transforms.contrast;
    float maxFactor = 1.0f + transforms.contrast;
    randomContrast(data, c, h, w, minFactor, maxFactor, rng);
  }

  // Gaussian noise (stddev from config, 0 = disabled)
  if (transforms.gaussianNoise > 0.0f) {
    std::bernoulli_distribution noiseCoin(0.3);
    if (noiseCoin(rng)) addGaussianNoise(data, transforms.gaussianNoise, rng);
  }
}

//===================================================================================================================//

} // namespace NN_CLI

