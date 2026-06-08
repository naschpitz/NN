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
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace NN_CLI
{

  //===================================================================================================================//

  std::vector<float> ImageLoader::loadImage(const std::string& imagePath, int targetC, int targetH, int targetW)
  {
    int origW = 0, origH = 0, origC = 0;
    unsigned char* pixels = stbi_load(imagePath.c_str(), &origW, &origH, &origC, targetC);

    if (!pixels) {
      throw std::runtime_error("Failed to load image: " + imagePath + " (" + stbi_failure_reason() + ")");
    }

    stbir_pixel_layout layout;

    if (targetC == 1)
      layout = STBIR_1CHEL;
    else if (targetC == 3)
      layout = STBIR_RGB;
    else if (targetC == 4)
      layout = STBIR_RGBA;
    else
      layout = STBIR_1CHEL; // fallback

    // Letterbox: scale to fit within target dimensions preserving aspect ratio, then centre on black canvas.
    float scaleH = static_cast<float>(targetH) / static_cast<float>(origH);
    float scaleW = static_cast<float>(targetW) / static_cast<float>(origW);
    float scale = std::min(scaleH, scaleW);

    int scaledW = static_cast<int>(std::round(origW * scale));
    int scaledH = static_cast<int>(std::round(origH * scale));

    // Clamp to target bounds (rounding may exceed by 1)
    scaledW = std::min(scaledW, targetW);
    scaledH = std::min(scaledH, targetH);

    // Resize the image to the scaled dimensions
    std::vector<unsigned char> scaledBuf(static_cast<size_t>(scaledW) * scaledH * targetC);

    if (origW != scaledW || origH != scaledH) {
      stbir_resize_uint8_linear(pixels, origW, origH, 0, scaledBuf.data(), scaledW, scaledH, 0, layout);
    } else {
      std::memcpy(scaledBuf.data(), pixels, scaledBuf.size());
    }

    stbi_image_free(pixels);

    // Create black canvas at target dimensions
    std::vector<unsigned char> canvas(static_cast<size_t>(targetW) * targetH * targetC, 0);

    // Centre the scaled image on the canvas
    int offsetY = (targetH - scaledH) / 2;
    int offsetX = (targetW - scaledW) / 2;
    int scaledStride = scaledW * targetC;
    int canvasStride = targetW * targetC;

    for (int y = 0; y < scaledH; ++y) {
      std::memcpy(canvas.data() + (offsetY + y) * canvasStride + offsetX * targetC, scaledBuf.data() + y * scaledStride,
                  static_cast<size_t>(scaledStride));
    }

    // Convert to flat NCHW float vector, normalised to [0, 1]
    std::vector<float> result(static_cast<size_t>(targetC) * targetH * targetW);

    for (int c = 0; c < targetC; ++c) {
      for (int h = 0; h < targetH; ++h) {
        for (int w = 0; w < targetW; ++w) {
          // Canvas is interleaved HWC: pixel[h * W * C + w * C + c]
          float val = static_cast<float>(canvas[h * canvasStride + w * targetC + c]) / 255.0f;
          // NCHW layout: data[c * H * W + h * W + w]
          result[c * targetH * targetW + h * targetW + w] = val;
        }
      }
    }

    return result;
  }

  //===================================================================================================================//

  void ImageLoader::saveImage(const std::string& imagePath, const std::vector<float>& data, int c, int h, int w)
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

  std::string ImageLoader::resolvePath(const std::string& imagePath, const std::string& baseDirPath)
  {
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

  void ImageLoader::horizontalFlip(std::vector<float>& data, int c, int h, int w)
  {
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

  void ImageLoader::randomRotation(std::vector<float>& data, int c, int h, int w, float maxDegrees, std::mt19937& rng)
  {
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
            if (sx < 0 || sx >= w || sy < 0 || sy >= h)
              return 0.0f;
            return data[chOffset + sy * w + sx];
          };

          result[chOffset + y * w + x] = (1 - fx) * (1 - fy) * sample(x0, y0) + fx * (1 - fy) * sample(x1, y0) +
                                         (1 - fx) * fy * sample(x0, y1) + fx * fy * sample(x1, y1);
        }
      }
    }

    data = std::move(result);
  }

  //===================================================================================================================//

  void ImageLoader::randomBrightness(std::vector<float>& data, int /*c*/, int /*h*/, int /*w*/, float maxDelta,
                                     std::mt19937& rng)
  {
    std::uniform_real_distribution<float> dist(-maxDelta, maxDelta);
    float delta = dist(rng);

    for (auto& v : data) {
      v = std::clamp(v + delta, 0.0f, 1.0f);
    }
  }

  //===================================================================================================================//

  void ImageLoader::randomContrast(std::vector<float>& data, int c, int h, int w, float minFactor, float maxFactor,
                                   std::mt19937& rng)
  {
    std::uniform_real_distribution<float> dist(minFactor, maxFactor);
    float factor = dist(rng);

    // Compute per-channel mean
    for (int ch = 0; ch < c; ch++) {
      int chOffset = ch * h * w;
      float mean = 0.0f;

      for (int i = 0; i < h * w; i++)
        mean += data[chOffset + i];
      mean /= static_cast<float>(h * w);

      for (int i = 0; i < h * w; i++) {
        data[chOffset + i] = std::clamp(mean + factor * (data[chOffset + i] - mean), 0.0f, 1.0f);
      }
    }
  }

  //===================================================================================================================//

  void ImageLoader::randomTranslation(std::vector<float>& data, int c, int h, int w, float maxFraction,
                                      std::mt19937& rng)
  {
    int maxDx = static_cast<int>(maxFraction * w);
    int maxDy = static_cast<int>(maxFraction * h);

    if (maxDx == 0 && maxDy == 0)
      return;

    std::uniform_int_distribution<int> distX(-maxDx, maxDx);
    std::uniform_int_distribution<int> distY(-maxDy, maxDy);
    int dx = distX(rng);
    int dy = distY(rng);

    std::vector<float> result(data.size(), 0.0f);

    for (int ch = 0; ch < c; ch++) {
      int chOffset = ch * h * w;

      for (int y = 0; y < h; y++) {
        int srcY = y - dy;

        if (srcY < 0 || srcY >= h)
          continue;

        for (int x = 0; x < w; x++) {
          int srcX = x - dx;

          if (srcX < 0 || srcX >= w)
            continue;
          result[chOffset + y * w + x] = data[chOffset + srcY * w + srcX];
        }
      }
    }

    data = std::move(result);
  }

  //===================================================================================================================//

  void ImageLoader::addGaussianNoise(std::vector<float>& data, float stddev, std::mt19937& rng)
  {
    std::normal_distribution<float> dist(0.0f, stddev);

    for (auto& v : data) {
      v = std::clamp(v + dist(rng), 0.0f, 1.0f);
    }
  }

  //===================================================================================================================//
  //-- Random Erasing (Cutout) --//
  //===================================================================================================================//

  void ImageLoader::randomErasing(std::vector<float>& data, int c, int h, int w, float maxArea, std::mt19937& rng)
  {
    float minArea = 0.02f;
    std::uniform_real_distribution<float> areaDist(minArea, maxArea);
    std::uniform_real_distribution<float> aspectDist(0.3f, 3.3f);

    float areaFrac = areaDist(rng);
    float aspect = aspectDist(rng);
    float imageArea = static_cast<float>(h * w);

    float eraseArea = areaFrac * imageArea;
    int eraseH = static_cast<int>(std::sqrt(eraseArea * aspect));
    int eraseW = static_cast<int>(std::sqrt(eraseArea / aspect));

    eraseH = std::min(eraseH, h);
    eraseW = std::min(eraseW, w);

    if (eraseH < 1 || eraseW < 1)
      return;

    std::uniform_int_distribution<int> yDist(0, h - eraseH);
    std::uniform_int_distribution<int> xDist(0, w - eraseW);

    int y0 = yDist(rng);
    int x0 = xDist(rng);

    for (int ch = 0; ch < c; ch++) {
      for (int y = y0; y < y0 + eraseH; y++) {
        for (int x = x0; x < x0 + eraseW; x++) {
          data[ch * h * w + y * w + x] = 0.0f;
        }
      }
    }
  }

  //===================================================================================================================//
  //-- Random Hue Shift --//
  //===================================================================================================================//

  void ImageLoader::randomHueShift(std::vector<float>& data, int c, int h, int w, float maxShift, std::mt19937& rng)
  {
    if (c != 3)
      return; // Hue shift only makes sense for RGB

    std::uniform_real_distribution<float> shiftDist(-maxShift, maxShift);
    float hueShift = shiftDist(rng) * 360.0f; // Convert fraction to degrees

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        float r = data[0 * h * w + y * w + x];
        float g = data[1 * h * w + y * w + x];
        float b = data[2 * h * w + y * w + x];

        // RGB → HSV
        float maxC = std::max({r, g, b});
        float minC = std::min({r, g, b});
        float delta = maxC - minC;

        float hue = 0.0f;
        float sat = (maxC > 0.0f) ? delta / maxC : 0.0f;
        float val = maxC;

        if (delta > 0.0f) {
          if (maxC == r)
            hue = 60.0f * std::fmod((g - b) / delta, 6.0f);
          else if (maxC == g)
            hue = 60.0f * ((b - r) / delta + 2.0f);
          else
            hue = 60.0f * ((r - g) / delta + 4.0f);
        }

        if (hue < 0.0f)
          hue += 360.0f;

        // Shift hue
        hue = std::fmod(hue + hueShift + 360.0f, 360.0f);

        // HSV → RGB
        float chroma = val * sat;
        float hPrime = hue / 60.0f;
        float xVal = chroma * (1.0f - std::fabs(std::fmod(hPrime, 2.0f) - 1.0f));
        float m = val - chroma;

        float r2 = 0.0f, g2 = 0.0f, b2 = 0.0f;

        if (hPrime < 1.0f) {
          r2 = chroma;
          g2 = xVal;
        } else if (hPrime < 2.0f) {
          r2 = xVal;
          g2 = chroma;
        } else if (hPrime < 3.0f) {
          g2 = chroma;
          b2 = xVal;
        } else if (hPrime < 4.0f) {
          g2 = xVal;
          b2 = chroma;
        } else if (hPrime < 5.0f) {
          r2 = xVal;
          b2 = chroma;
        } else {
          r2 = chroma;
          b2 = xVal;
        }

        data[0 * h * w + y * w + x] = std::clamp(r2 + m, 0.0f, 1.0f);
        data[1 * h * w + y * w + x] = std::clamp(g2 + m, 0.0f, 1.0f);
        data[2 * h * w + y * w + x] = std::clamp(b2 + m, 0.0f, 1.0f);
      }
    }
  }

  //===================================================================================================================//
  //-- Random Scaling (Zoom) --//
  //===================================================================================================================//

  void ImageLoader::randomScaling(std::vector<float>& data, int c, int h, int w, float maxScale, std::mt19937& rng)
  {
    std::uniform_real_distribution<float> scaleDist(1.0f - maxScale, 1.0f + maxScale);
    float scale = scaleDist(rng);

    if (std::fabs(scale - 1.0f) < 0.01f)
      return; // No meaningful change

    int newH = static_cast<int>(std::round(h * scale));
    int newW = static_cast<int>(std::round(w * scale));

    if (newH < 1 || newW < 1)
      return;

    // Resize using bilinear interpolation
    std::vector<float> resized(c * newH * newW, 0.0f);

    for (int ch = 0; ch < c; ch++) {
      for (int y = 0; y < newH; y++) {
        for (int x = 0; x < newW; x++) {
          float srcY = static_cast<float>(y) * h / newH;
          float srcX = static_cast<float>(x) * w / newW;

          int y0 = static_cast<int>(srcY);
          int x0 = static_cast<int>(srcX);
          int y1 = std::min(y0 + 1, h - 1);
          int x1 = std::min(x0 + 1, w - 1);

          float fy = srcY - y0;
          float fx = srcX - x0;

          float v00 = data[ch * h * w + y0 * w + x0];
          float v10 = data[ch * h * w + y1 * w + x0];
          float v01 = data[ch * h * w + y0 * w + x1];
          float v11 = data[ch * h * w + y1 * w + x1];

          resized[ch * newH * newW + y * newW + x] =
            v00 * (1 - fy) * (1 - fx) + v10 * fy * (1 - fx) + v01 * (1 - fy) * fx + v11 * fy * fx;
        }
      }
    }

    // Center crop (zoom in) or center pad (zoom out) back to original size
    std::fill(data.begin(), data.end(), 0.0f);

    int offsetY = (newH - h) / 2;
    int offsetX = (newW - w) / 2;

    for (int ch = 0; ch < c; ch++) {
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          int srcY = y + offsetY;
          int srcX = x + offsetX;

          if (srcY >= 0 && srcY < newH && srcX >= 0 && srcX < newW) {
            data[ch * h * w + y * w + x] = resized[ch * newH * newW + srcY * newW + srcX];
          }
        }
      }
    }
  }

  //===================================================================================================================//
  //-- Elastic Deformation --//
  //===================================================================================================================//

  void ImageLoader::elasticDeformation(std::vector<float>& data, int c, int h, int w, float alpha, float sigma,
                                       std::mt19937& rng)
  {
    // Generate random displacement fields
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> dx(h * w);
    std::vector<float> dy(h * w);

    for (int i = 0; i < h * w; i++) {
      dx[i] = dist(rng);
      dy[i] = dist(rng);
    }

    // Gaussian smooth the displacement fields
    int kernelRadius = static_cast<int>(std::ceil(sigma * 3.0f));
    std::vector<float> kernel(2 * kernelRadius + 1);
    float kernelSum = 0.0f;

    for (int i = -kernelRadius; i <= kernelRadius; i++) {
      kernel[i + kernelRadius] = std::exp(-0.5f * (i * i) / (sigma * sigma));
      kernelSum += kernel[i + kernelRadius];
    }

    for (auto& k : kernel)
      k /= kernelSum;

    // Separable Gaussian: horizontal pass
    auto smoothH = [&](std::vector<float>& field) {
      std::vector<float> tmp(h * w, 0.0f);

      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          float val = 0.0f;

          for (int k = -kernelRadius; k <= kernelRadius; k++) {
            int sx = std::clamp(x + k, 0, w - 1);
            val += field[y * w + sx] * kernel[k + kernelRadius];
          }

          tmp[y * w + x] = val;
        }
      }

      field = tmp;
    };

    // Separable Gaussian: vertical pass
    auto smoothV = [&](std::vector<float>& field) {
      std::vector<float> tmp(h * w, 0.0f);

      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          float val = 0.0f;

          for (int k = -kernelRadius; k <= kernelRadius; k++) {
            int sy = std::clamp(y + k, 0, h - 1);
            val += field[sy * w + x] * kernel[k + kernelRadius];
          }

          tmp[y * w + x] = val;
        }
      }

      field = tmp;
    };

    smoothH(dx);
    smoothV(dx);
    smoothH(dy);
    smoothV(dy);

    // Scale by alpha
    for (int i = 0; i < h * w; i++) {
      dx[i] *= alpha;
      dy[i] *= alpha;
    }

    // Apply deformation with bilinear interpolation
    std::vector<float> result(data.size(), 0.0f);

    for (int ch = 0; ch < c; ch++) {
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          float srcY = y + dy[y * w + x];
          float srcX = x + dx[y * w + x];

          int y0 = static_cast<int>(std::floor(srcY));
          int x0 = static_cast<int>(std::floor(srcX));
          int y1 = y0 + 1;
          int x1 = x0 + 1;

          y0 = std::clamp(y0, 0, h - 1);
          y1 = std::clamp(y1, 0, h - 1);
          x0 = std::clamp(x0, 0, w - 1);
          x1 = std::clamp(x1, 0, w - 1);

          float fy = srcY - std::floor(srcY);
          float fx = srcX - std::floor(srcX);

          float v00 = data[ch * h * w + y0 * w + x0];
          float v10 = data[ch * h * w + y1 * w + x0];
          float v01 = data[ch * h * w + y0 * w + x1];
          float v11 = data[ch * h * w + y1 * w + x1];

          result[ch * h * w + y * w + x] =
            v00 * (1 - fy) * (1 - fx) + v10 * fy * (1 - fx) + v01 * (1 - fy) * fx + v11 * fy * fx;
        }
      }
    }

    data = std::move(result);
  }

  //===================================================================================================================//

  void ImageLoader::applyRandomTransforms(std::vector<float>& data, int c, int h, int w, std::mt19937& rng,
                                          const AugmentationTransforms& transforms, float probability)
  {
    std::bernoulli_distribution coin(probability);

    // Geometric transforms first
    if (transforms.scaling > 0.0f && coin(rng))
      randomScaling(data, c, h, w, transforms.scaling, rng);

    if (transforms.elasticDeformation.alpha > 0.0f && coin(rng))
      elasticDeformation(data, c, h, w, transforms.elasticDeformation.alpha, transforms.elasticDeformation.sigma, rng);

    if (transforms.horizontalFlip && coin(rng))
      horizontalFlip(data, c, h, w);

    if (transforms.rotation > 0.0f && coin(rng))
      randomRotation(data, c, h, w, transforms.rotation, rng);

    if (transforms.translation > 0.0f && coin(rng))
      randomTranslation(data, c, h, w, transforms.translation, rng);

    // Color transforms
    if (transforms.hueShift > 0.0f && coin(rng))
      randomHueShift(data, c, h, w, transforms.hueShift, rng);

    if (transforms.brightness > 0.0f && coin(rng))
      randomBrightness(data, c, h, w, transforms.brightness, rng);

    if (transforms.contrast > 0.0f && coin(rng)) {
      float minFactor = 1.0f - transforms.contrast;
      float maxFactor = 1.0f + transforms.contrast;
      randomContrast(data, c, h, w, minFactor, maxFactor, rng);
    }

    // Destructive transforms last
    if (transforms.randomErasing > 0.0f && coin(rng))
      randomErasing(data, c, h, w, transforms.randomErasing, rng);

    if (transforms.gaussianNoise > 0.0f && coin(rng))
      addGaussianNoise(data, transforms.gaussianNoise, rng);
  }

  //===================================================================================================================//

} // namespace NN_CLI
