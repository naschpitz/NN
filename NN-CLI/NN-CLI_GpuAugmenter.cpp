#include "NN-CLI_GpuAugmenter.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

namespace NN_CLI
{
  //===================================================================================================================//
  //-- GpuAugmenter --//
  //===================================================================================================================//

  GpuAugmenter::GpuAugmenter(int deviceIndex, ulong c, ulong h, ulong w, LogLevel logLevel)
    : core(deviceIndex),
      C(c),
      H(h),
      W(w),
      logLevel(logLevel)
  {
    // Sub-batch size: process the (possibly large) host batch in GPU-sized chunks so
    // the device buffers stay bounded (~256 MB per image buffer).
    ulong elemsPerImage = std::max<ulong>(1, this->C * this->H * this->W);
    ulong targetElems = 64UL * 1024 * 1024; // ~64M floats ≈ 256 MB
    this->capacity = std::max<ulong>(1, std::min<ulong>(targetElems / elemsPerImage, 512));

    // Load the augmentation kernel source (path relative to this .cpp, like CNN does).
    std::string srcFile = __FILE__;
    std::string srcDir = srcFile.substr(0, srcFile.find_last_of("/\\") + 1);
    this->core.addSourceFile(srcDir + "opencl/NN-CLI_Augment.cpp.cl");

    ulong cap = this->capacity;
    ulong chw = this->C * this->H * this->W;
    ulong hw = this->H * this->W;

    this->core.allocateBuffer<float>("aug_a", cap * chw);
    this->core.allocateBuffer<float>("aug_b", cap * chw);
    this->core.allocateBuffer<float>("aug_means", cap * this->C);
    this->core.allocateBuffer<float>("aug_dxField", cap * hw);
    this->core.allocateBuffer<float>("aug_dyField", cap * hw);
    this->core.allocateBuffer<float>("aug_fieldTmp", cap * hw); // scratch for separable blur
    this->core.allocateBuffer<float>("aug_gauss", 257); // 1D Gaussian kernel (radius <= 128)

    for (const char* name : {"aug_applyFlip", "aug_applyRot", "aug_applyTrans", "aug_applyScale", "aug_applyElastic",
                             "aug_applyHue", "aug_applyBri", "aug_applyCon", "aug_applyErase", "aug_applyNoise",
                             "aug_dxs", "aug_dys", "aug_eraseX0", "aug_eraseY0", "aug_eraseW", "aug_eraseH"})
      this->core.allocateBuffer<int>(name, cap);

    for (const char* name : {"aug_angleRad", "aug_scales", "aug_shiftDeg", "aug_deltas", "aug_factors"})
      this->core.allocateBuffer<float>(name, cap);

    // Host staging vectors.
    this->applyFlip.resize(cap);
    this->applyRot.resize(cap);
    this->applyTrans.resize(cap);
    this->applyScale.resize(cap);
    this->applyElastic.resize(cap);
    this->applyHue.resize(cap);
    this->applyBri.resize(cap);
    this->applyCon.resize(cap);
    this->applyErase.resize(cap);
    this->applyNoise.resize(cap);
    this->angleRad.resize(cap);
    this->scales.resize(cap);
    this->shiftDeg.resize(cap);
    this->deltas.resize(cap);
    this->factors.resize(cap);
    this->dxs.resize(cap);
    this->dys.resize(cap);
    this->eraseX0.resize(cap);
    this->eraseY0.resize(cap);
    this->eraseW.resize(cap);
    this->eraseH.resize(cap);

    if (this->logLevel >= LogLevel::INFO)
      std::cout << "GpuAugmenter on device " << deviceIndex << " (sub-batch " << this->capacity << ").\n";
  }

  //===================================================================================================================//

  void GpuAugmenter::augment(std::vector<float>& batch, ulong count, const AugmentationTransforms& t, float probability,
                             std::mt19937& rng)
  {
    if (count == 0)
      return;

    const ulong C = this->C, H = this->H, W = this->W;
    const ulong chw = C * H * W, hw = H * W;

    // Which transforms are active for this run.
    const bool eScale = t.scaling > 0.0f;
    const bool eElastic = t.elasticDeformation.alpha > 0.0f;
    const bool eFlip = t.horizontalFlip;
    const bool eRot = t.rotation > 0.0f;
    const bool eTrans = t.translation > 0.0f;
    const bool eHue = t.hueShift > 0.0f && C == 3;
    const bool eBri = t.brightness > 0.0f;
    const bool eCon = t.contrast > 0.0f;
    const bool eErase = t.randomErasing > 0.0f;
    const bool eNoise = t.gaussianNoise > 0.0f;

    if (!(eScale || eElastic || eFlip || eRot || eTrans || eHue || eBri || eCon || eErase || eNoise))
      return; // nothing to do

    std::bernoulli_distribution coin(probability);

    for (ulong baseSample = 0; baseSample < count; baseSample += this->capacity) {
      ulong n = std::min(this->capacity, count - baseSample);
      ulong total = n * chw;
      int elasticRadius = 0; // set when elastic is enabled (used by the field-blur kernels)

      // ---- draw per-sample parameters (host RNG, matching the CPU transforms) ---- //
      for (ulong i = 0; i < n; i++) {
        if (eScale) {
          this->applyScale[i] = 0;
          this->scales[i] = 1.0f;

          if (coin(rng)) {
            float s = std::uniform_real_distribution<float>(1.0f - t.scaling, 1.0f + t.scaling)(rng);

            if (std::fabs(s - 1.0f) >= 0.01f) {
              this->applyScale[i] = 1;
              this->scales[i] = s;
            }
          }
        }

        if (eElastic)
          this->applyElastic[i] = coin(rng) ? 1 : 0;

        if (eFlip)
          this->applyFlip[i] = coin(rng) ? 1 : 0;

        if (eRot) {
          this->applyRot[i] = coin(rng) ? 1 : 0;
          this->angleRad[i] = this->applyRot[i] ? std::uniform_real_distribution<float>(-t.rotation, t.rotation)(rng) *
                                                    3.14159265f / 180.0f
                                                : 0.0f;
        }

        if (eTrans) {
          this->applyTrans[i] = coin(rng) ? 1 : 0;
          int maxDx = static_cast<int>(t.translation * static_cast<float>(W));
          int maxDy = static_cast<int>(t.translation * static_cast<float>(H));
          this->dxs[i] =
            (this->applyTrans[i] && maxDx > 0) ? std::uniform_int_distribution<int>(-maxDx, maxDx)(rng) : 0;
          this->dys[i] =
            (this->applyTrans[i] && maxDy > 0) ? std::uniform_int_distribution<int>(-maxDy, maxDy)(rng) : 0;
        }

        if (eHue) {
          this->applyHue[i] = coin(rng) ? 1 : 0;
          this->shiftDeg[i] =
            this->applyHue[i] ? std::uniform_real_distribution<float>(-t.hueShift, t.hueShift)(rng) * 360.0f : 0.0f;
        }

        if (eBri) {
          this->applyBri[i] = coin(rng) ? 1 : 0;
          this->deltas[i] =
            this->applyBri[i] ? std::uniform_real_distribution<float>(-t.brightness, t.brightness)(rng) : 0.0f;
        }

        if (eCon) {
          this->applyCon[i] = coin(rng) ? 1 : 0;
          this->factors[i] =
            this->applyCon[i] ? std::uniform_real_distribution<float>(1.0f - t.contrast, 1.0f + t.contrast)(rng) : 1.0f;
        }

        if (eErase) {
          this->applyErase[i] = coin(rng) ? 1 : 0;

          if (this->applyErase[i]) {
            float areaFrac = std::uniform_real_distribution<float>(0.02f, t.randomErasing)(rng);
            float aspect = std::uniform_real_distribution<float>(0.3f, 3.3f)(rng);
            float area = areaFrac * static_cast<float>(H) * static_cast<float>(W);
            long eh = std::min(static_cast<long>(std::round(std::sqrt(area * aspect))), static_cast<long>(H));
            long ew = std::min(static_cast<long>(std::round(std::sqrt(area / aspect))), static_cast<long>(W));
            this->eraseH[i] = static_cast<int>(eh);
            this->eraseW[i] = static_cast<int>(ew);
            this->eraseY0[i] =
              (static_cast<long>(H) > eh)
                ? std::uniform_int_distribution<int>(0, static_cast<int>(static_cast<long>(H) - eh))(rng)
                : 0;
            this->eraseX0[i] =
              (static_cast<long>(W) > ew)
                ? std::uniform_int_distribution<int>(0, static_cast<int>(static_cast<long>(W) - ew))(rng)
                : 0;
          } else {
            this->eraseH[i] = this->eraseW[i] = this->eraseX0[i] = this->eraseY0[i] = 0;
          }
        }

        if (eNoise)
          this->applyNoise[i] = coin(rng) ? 1 : 0;
      }

      // ---- upload image slice + parameters ---- //
      std::vector<float> subImg(batch.begin() + baseSample * chw, batch.begin() + (baseSample + n) * chw);
      this->core.writeBuffer<float>("aug_a", subImg, 0);

      if (eScale) {
        this->core.writeBuffer<int>("aug_applyScale", this->applyScale, 0);
        this->core.writeBuffer<float>("aug_scales", this->scales, 0);
      }

      if (eElastic) {
        this->core.writeBuffer<int>("aug_applyElastic", this->applyElastic, 0);

        // Build the 1D Gaussian kernel on the host (cheap, small) and upload it.
        float sigma = t.elasticDeformation.sigma;
        elasticRadius = std::min(128, std::max(1, static_cast<int>(std::ceil(sigma * 3.0f))));
        std::vector<float> gauss(2 * elasticRadius + 1);
        float gsum = 0.0f;

        for (int k = -elasticRadius; k <= elasticRadius; k++) {
          float v = std::exp(-0.5f * static_cast<float>(k * k) / (sigma * sigma));
          gauss[k + elasticRadius] = v;
          gsum += v;
        }

        for (float& v : gauss)
          v /= gsum;

        this->core.writeBuffer<float>("aug_gauss", gauss, 0);
      }

      if (eFlip)
        this->core.writeBuffer<int>("aug_applyFlip", this->applyFlip, 0);

      if (eRot) {
        this->core.writeBuffer<int>("aug_applyRot", this->applyRot, 0);
        this->core.writeBuffer<float>("aug_angleRad", this->angleRad, 0);
      }

      if (eTrans) {
        this->core.writeBuffer<int>("aug_applyTrans", this->applyTrans, 0);
        this->core.writeBuffer<int>("aug_dxs", this->dxs, 0);
        this->core.writeBuffer<int>("aug_dys", this->dys, 0);
      }

      if (eHue) {
        this->core.writeBuffer<int>("aug_applyHue", this->applyHue, 0);
        this->core.writeBuffer<float>("aug_shiftDeg", this->shiftDeg, 0);
      }

      if (eBri) {
        this->core.writeBuffer<int>("aug_applyBri", this->applyBri, 0);
        this->core.writeBuffer<float>("aug_deltas", this->deltas, 0);
      }

      if (eCon) {
        this->core.writeBuffer<int>("aug_applyCon", this->applyCon, 0);
        this->core.writeBuffer<float>("aug_factors", this->factors, 0);
      }

      if (eErase) {
        this->core.writeBuffer<int>("aug_applyErase", this->applyErase, 0);
        this->core.writeBuffer<int>("aug_eraseX0", this->eraseX0, 0);
        this->core.writeBuffer<int>("aug_eraseY0", this->eraseY0, 0);
        this->core.writeBuffer<int>("aug_eraseW", this->eraseW, 0);
        this->core.writeBuffer<int>("aug_eraseH", this->eraseH, 0);
      }

      if (eNoise)
        this->core.writeBuffer<int>("aug_applyNoise", this->applyNoise, 0);

      // ---- build kernel sequence (geometric ping-pong, then element-wise in place) ---- //
      this->core.clearKernels();
      std::string cur = "aug_a", other = "aug_b";

      if (eScale) {
        this->core.addKernel("k_scale", "aug_scale", total, 0);
        this->core.addArgument<float>("k_scale", cur);
        this->core.addArgument<float>("k_scale", other);
        this->core.addArgument<int>("k_scale", "aug_applyScale");
        this->core.addArgument<float>("k_scale", "aug_scales");
        this->core.addArgument<ulong>("k_scale", C);
        this->core.addArgument<ulong>("k_scale", H);
        this->core.addArgument<ulong>("k_scale", W);
        this->core.addArgument<ulong>("k_scale", total);
        std::swap(cur, other);
      }

      if (eElastic) {
        ulong fieldTotal = n * hw;
        unsigned int fseed = static_cast<unsigned int>(rng());

        // Generate the displacement field on the GPU: random fill, then a separable
        // Gaussian blur (horizontal -> tmp, vertical -> field) for dx and dy.
        this->core.addKernel("k_field_rand", "aug_field_random", fieldTotal, 0);
        this->core.addArgument<float>("k_field_rand", "aug_dxField");
        this->core.addArgument<float>("k_field_rand", "aug_dyField");
        this->core.addArgument<unsigned int>("k_field_rand", fseed);
        this->core.addArgument<ulong>("k_field_rand", fieldTotal);

        const char* blurStages[4][3] = {{"k_blur_dxh", "aug_dxField", "aug_fieldTmp"},
                                        {"k_blur_dxv", "aug_fieldTmp", "aug_dxField"},
                                        {"k_blur_dyh", "aug_dyField", "aug_fieldTmp"},
                                        {"k_blur_dyv", "aug_fieldTmp", "aug_dyField"}};

        for (int b = 0; b < 4; b++) {
          int horizontal = (b % 2 == 0) ? 1 : 0;
          this->core.addKernel(blurStages[b][0], "aug_field_blur", fieldTotal, 0);
          this->core.addArgument<float>(blurStages[b][0], blurStages[b][1]);
          this->core.addArgument<float>(blurStages[b][0], blurStages[b][2]);
          this->core.addArgument<float>(blurStages[b][0], "aug_gauss");
          this->core.addArgument<int>(blurStages[b][0], elasticRadius);
          this->core.addArgument<ulong>(blurStages[b][0], H);
          this->core.addArgument<ulong>(blurStages[b][0], W);
          this->core.addArgument<int>(blurStages[b][0], horizontal);
          this->core.addArgument<ulong>(blurStages[b][0], fieldTotal);
        }

        this->core.addKernel("k_elastic", "aug_elastic_apply", total, 0);
        this->core.addArgument<float>("k_elastic", cur);
        this->core.addArgument<float>("k_elastic", other);
        this->core.addArgument<int>("k_elastic", "aug_applyElastic");
        this->core.addArgument<float>("k_elastic", "aug_dxField");
        this->core.addArgument<float>("k_elastic", "aug_dyField");
        this->core.addArgument<float>("k_elastic", t.elasticDeformation.alpha);
        this->core.addArgument<ulong>("k_elastic", C);
        this->core.addArgument<ulong>("k_elastic", H);
        this->core.addArgument<ulong>("k_elastic", W);
        this->core.addArgument<ulong>("k_elastic", total);
        std::swap(cur, other);
      }

      if (eFlip) {
        this->core.addKernel("k_flip", "aug_flip", total, 0);
        this->core.addArgument<float>("k_flip", cur);
        this->core.addArgument<float>("k_flip", other);
        this->core.addArgument<int>("k_flip", "aug_applyFlip");
        this->core.addArgument<ulong>("k_flip", C);
        this->core.addArgument<ulong>("k_flip", H);
        this->core.addArgument<ulong>("k_flip", W);
        this->core.addArgument<ulong>("k_flip", total);
        std::swap(cur, other);
      }

      if (eRot) {
        this->core.addKernel("k_rot", "aug_rotate", total, 0);
        this->core.addArgument<float>("k_rot", cur);
        this->core.addArgument<float>("k_rot", other);
        this->core.addArgument<int>("k_rot", "aug_applyRot");
        this->core.addArgument<float>("k_rot", "aug_angleRad");
        this->core.addArgument<ulong>("k_rot", C);
        this->core.addArgument<ulong>("k_rot", H);
        this->core.addArgument<ulong>("k_rot", W);
        this->core.addArgument<ulong>("k_rot", total);
        std::swap(cur, other);
      }

      if (eTrans) {
        this->core.addKernel("k_trans", "aug_translate", total, 0);
        this->core.addArgument<float>("k_trans", cur);
        this->core.addArgument<float>("k_trans", other);
        this->core.addArgument<int>("k_trans", "aug_applyTrans");
        this->core.addArgument<int>("k_trans", "aug_dxs");
        this->core.addArgument<int>("k_trans", "aug_dys");
        this->core.addArgument<ulong>("k_trans", C);
        this->core.addArgument<ulong>("k_trans", H);
        this->core.addArgument<ulong>("k_trans", W);
        this->core.addArgument<ulong>("k_trans", total);
        std::swap(cur, other);
      }

      // Element-wise transforms operate in place on `cur`.
      if (eHue) {
        ulong totalPixels = n * hw;
        this->core.addKernel("k_hue", "aug_hue", totalPixels, 0);
        this->core.addArgument<float>("k_hue", cur);
        this->core.addArgument<int>("k_hue", "aug_applyHue");
        this->core.addArgument<float>("k_hue", "aug_shiftDeg");
        this->core.addArgument<ulong>("k_hue", H);
        this->core.addArgument<ulong>("k_hue", W);
        this->core.addArgument<ulong>("k_hue", totalPixels);
      }

      if (eBri) {
        this->core.addKernel("k_bri", "aug_brightness", total, 0);
        this->core.addArgument<float>("k_bri", cur);
        this->core.addArgument<int>("k_bri", "aug_applyBri");
        this->core.addArgument<float>("k_bri", "aug_deltas");
        this->core.addArgument<ulong>("k_bri", C);
        this->core.addArgument<ulong>("k_bri", H);
        this->core.addArgument<ulong>("k_bri", W);
        this->core.addArgument<ulong>("k_bri", total);
      }

      if (eCon) {
        this->core.addKernel("k_mean", "aug_channel_mean", n * C * 256, 0, 256);
        this->core.addArgument<float>("k_mean", cur);
        this->core.addArgument<float>("k_mean", "aug_means");
        this->core.addArgument<ulong>("k_mean", C);
        this->core.addArgument<ulong>("k_mean", H);
        this->core.addArgument<ulong>("k_mean", W);

        this->core.addKernel("k_con", "aug_contrast", total, 0);
        this->core.addArgument<float>("k_con", cur);
        this->core.addArgument<int>("k_con", "aug_applyCon");
        this->core.addArgument<float>("k_con", "aug_factors");
        this->core.addArgument<float>("k_con", "aug_means");
        this->core.addArgument<ulong>("k_con", C);
        this->core.addArgument<ulong>("k_con", H);
        this->core.addArgument<ulong>("k_con", W);
        this->core.addArgument<ulong>("k_con", total);
      }

      if (eErase) {
        this->core.addKernel("k_erase", "aug_erase", total, 0);
        this->core.addArgument<float>("k_erase", cur);
        this->core.addArgument<int>("k_erase", "aug_applyErase");
        this->core.addArgument<int>("k_erase", "aug_eraseX0");
        this->core.addArgument<int>("k_erase", "aug_eraseY0");
        this->core.addArgument<int>("k_erase", "aug_eraseW");
        this->core.addArgument<int>("k_erase", "aug_eraseH");
        this->core.addArgument<ulong>("k_erase", C);
        this->core.addArgument<ulong>("k_erase", H);
        this->core.addArgument<ulong>("k_erase", W);
        this->core.addArgument<ulong>("k_erase", total);
      }

      if (eNoise) {
        unsigned int seed = static_cast<unsigned int>(rng());
        this->core.addKernel("k_noise", "aug_gaussian_noise", total, 0);
        this->core.addArgument<float>("k_noise", cur);
        this->core.addArgument<int>("k_noise", "aug_applyNoise");
        this->core.addArgument<float>("k_noise", t.gaussianNoise);
        this->core.addArgument<unsigned int>("k_noise", seed);
        this->core.addArgument<ulong>("k_noise", C);
        this->core.addArgument<ulong>("k_noise", H);
        this->core.addArgument<ulong>("k_noise", W);
        this->core.addArgument<ulong>("k_noise", total);
      }

      this->core.run();

      // ---- download the augmented slice back into the batch ---- //
      this->core.readBuffer<float>(cur, subImg, 0);
      std::copy(subImg.begin(), subImg.end(), batch.begin() + baseSample * chw);
    }
  }

  //===================================================================================================================//
  //-- GpuAugmenterPool --//
  //===================================================================================================================//

  GpuAugmenterPool::GpuAugmenterPool(const std::vector<int>& deviceIndices, ulong c, ulong h, ulong w,
                                     LogLevel logLevel)
  {
    for (size_t i = 0; i < deviceIndices.size(); i++) {
      this->augmenters.push_back(std::make_unique<GpuAugmenter>(deviceIndices[i], c, h, w, logLevel));
      this->rngs.push_back(std::make_unique<std::mt19937>(0xA06 + deviceIndices[i]));
      this->freeList.push_back(static_cast<int>(i));
    }
  }

  //===================================================================================================================//

  void GpuAugmenterPool::augment(std::vector<float>& batch, ulong count, const AugmentationTransforms& transforms,
                                 float probability)
  {
    if (this->timingCallback)
      this->timingCallback(true);

    // Acquire a free augmenter (blocks until one is available).
    int idx;
    {
      QMutexLocker<QMutex> lock(&this->mutex);

      while (this->freeList.empty())
        this->cv.wait(&this->mutex);

      idx = this->freeList.back();
      this->freeList.pop_back();
    }

    // Return the augmenter to the free list even when augment() throws —
    // leaking the slot would leave every later caller blocked forever on
    // the wait above.
    auto release = [this, idx]() {
      {
        QMutexLocker<QMutex> lock(&this->mutex);
        this->freeList.push_back(idx);
      }

      this->cv.wakeOne();
    };

    try {
      this->augmenters[idx]->augment(batch, count, transforms, probability, *this->rngs[idx]);
    } catch (...) {
      release();
      throw;
    }

    release();

    if (this->timingCallback)
      this->timingCallback(false);
  }
}
