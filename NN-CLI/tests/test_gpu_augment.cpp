#include "test_helpers.hpp"

#include "NN-CLI_GpuAugmenter.hpp"

#include <OCLW_Core.hpp>

#include <chrono>
#include <cmath>
#include <random>
#include <vector>

using namespace NN_CLI;

//===================================================================================================================//
// GPU augmentation unit tests — exercise the OpenCL augmentation kernels directly on
// small images, validating that they compile, run, and produce correct output.
//===================================================================================================================//

static AugmentationTransforms allDisabled()
{
  AugmentationTransforms t;
  t.horizontalFlip = false;
  t.rotation = 0.0f;
  t.translation = 0.0f;
  t.brightness = 0.0f;
  t.contrast = 0.0f;
  t.gaussianNoise = 0.0f;
  t.randomErasing = 0.0f;
  t.hueShift = 0.0f;
  t.scaling = 0.0f;
  t.elasticDeformation.alpha = 0.0f;
  return t;
}

void runGpuAugmentTests()
{
  std::cout << "\n--- GPU Augmentation Tests ---" << std::endl;

  OpenCLWrapper::Core::initialize(false);

  if (OpenCLWrapper::Core::getNumDevices() == 0) {
    std::cout << "  (no GPU device available — skipping)" << std::endl;
    return;
  }

  std::mt19937 rng(12345);

  // 1. Horizontal flip is deterministic with probability 1.0 — mirror along width.
  {
    GpuAugmenter aug(0, 1, 2, 3, LogLevel::QUIET);
    AugmentationTransforms t = allDisabled();
    t.horizontalFlip = true;

    std::vector<float> img = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    aug.augment(img, 1, t, 1.0f, rng);

    std::vector<float> expected = {0.3f, 0.2f, 0.1f, 0.6f, 0.5f, 0.4f};
    bool ok = true;

    for (size_t i = 0; i < img.size(); i++)

      if (std::fabs(img[i] - expected[i]) > 1e-4f)
        ok = false;
    CHECK(ok, "GPU flip mirrors the image along width");
  }

  // 2. With no transforms enabled, augment is a no-op.
  {
    GpuAugmenter aug(0, 3, 4, 4, LogLevel::QUIET);
    AugmentationTransforms t = allDisabled();

    std::vector<float> img(3 * 4 * 4);

    for (size_t i = 0; i < img.size(); i++)
      img[i] = (float)i / (float)img.size();
    std::vector<float> orig = img;

    aug.augment(img, 1, t, 1.0f, rng);

    bool ok = true;

    for (size_t i = 0; i < img.size(); i++)

      if (std::fabs(img[i] - orig[i]) > 1e-6f)
        ok = false;
    CHECK(ok, "GPU augment is a no-op when all transforms are disabled");
  }

  // 3. Brightness adds a single per-sample delta — a uniform image stays uniform and in range.
  {
    GpuAugmenter aug(0, 1, 4, 4, LogLevel::QUIET);
    AugmentationTransforms t = allDisabled();
    t.brightness = 0.1f;

    std::vector<float> img(16, 0.5f);
    aug.augment(img, 1, t, 1.0f, rng);

    bool uniform = true, inRange = true;

    for (float v : img) {
      if (std::fabs(v - img[0]) > 1e-4f)
        uniform = false;

      if (v < 0.39f || v > 0.61f)
        inRange = false;
    }

    CHECK(uniform, "GPU brightness keeps a uniform image uniform");
    CHECK(inRange, "GPU brightness stays in [0.4, 0.6] for a 0.5 input (delta in +/-0.1)");
  }

  // 4. The full augmentation set keeps output finite and within [0, 1] (multi-sample batch).
  {
    const ulong C = 3, H = 16, W = 16, N = 5;
    GpuAugmenter aug(0, C, H, W, LogLevel::QUIET);
    AugmentationTransforms t; // defaults: flip, rotation, translation, brightness, contrast, noise
    t.randomErasing = 0.25f;
    t.hueShift = 0.05f;
    t.scaling = 0.15f;
    t.elasticDeformation.alpha = 20.0f;

    std::vector<float> img(N * C * H * W);

    for (size_t i = 0; i < img.size(); i++)
      img[i] = (float)((i * 37) % 100) / 100.0f;
    std::vector<float> orig = img;

    aug.augment(img, N, t, 1.0f, rng);

    bool ok = true, changed = false;

    for (size_t i = 0; i < img.size(); i++) {
      if (!std::isfinite(img[i]) || img[i] < -1e-4f || img[i] > 1.0001f)
        ok = false;

      if (std::fabs(img[i] - orig[i]) > 1e-4f)
        changed = true;
    }

    CHECK(ok, "GPU full augmentation output is finite and within [0, 1]");
    CHECK(changed, "GPU full augmentation actually modifies the batch");
  }

  // 5. Throughput probe at a realistic image size — full set vs without elastic.
  // (Informational: helps locate the bottleneck. Also asserts it completes.)
  {
    const ulong C = 3, H = 384, W = 384, N = 32;
    GpuAugmenter aug(0, C, H, W, LogLevel::QUIET);

    std::vector<float> img(N * C * H * W);

    for (size_t i = 0; i < img.size(); i++)
      img[i] = (float)((i * 37) % 100) / 100.0f;

    auto timeRun = [&](const AugmentationTransforms& t, const char* label) {
      std::vector<float> b = img;
      aug.augment(b, N, t, 1.0f, rng); // warm up (compiles kernels on first run)
      auto t0 = std::chrono::steady_clock::now();

      for (int r = 0; r < 5; r++) {
        b = img;
        aug.augment(b, N, t, 1.0f, rng);
      }

      double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::cout << "  [perf] " << label << ": " << (int)(5 * N / dt) << " img/s" << std::endl;
    };

    AugmentationTransforms full; // defaults
    full.randomErasing = 0.25f;
    full.hueShift = 0.05f;
    full.scaling = 0.15f;
    full.elasticDeformation.alpha = 30.0f;
    timeRun(full, "full (with elastic)");

    AugmentationTransforms noElastic = full;
    noElastic.elasticDeformation.alpha = 0.0f;
    timeRun(noElastic, "without elastic");

    CHECK(true, "GPU augmentation throughput probe completed");
  }

  std::cout << "GPU Augmentation Tests done." << std::endl;
}
