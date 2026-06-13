#ifndef NN_CLI_CALIBRATEUTILS_HPP
#define NN_CLI_CALIBRATEUTILS_HPP

#include "NN-CLI_IOConfig.hpp"
#include "NN-CLI_ImageLoader.hpp"
#include "NN-CLI_LogLevel.hpp"
#include "NN-CLI_Utils.hpp"

#include <ANN_Core.hpp>
#include <CNN_Core.hpp>

#include <json.hpp>

#include <QProcess>
#include <QString>
#include <QStringList>
#include <QVector>
#include <QtConcurrent>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <functional>
#include <random>
#include <string>
#include <vector>

//===================================================================================================================//

namespace NN_CLI
{

  //---- CalibrationConfig ------------------------------------------------------//

  /**
   * Configuration for the calibration pipeline.
   *
   * Controls which image directories are used, how many samples are drawn,
   * which ID percentile becomes the OOD threshold, and where the result
   * JSON is written.
   */
  struct CalibrationConfig {
    std::string idImagesDir;
    std::string oodDir;
    std::size_t idSampleCount = 500;
    std::size_t oodSampleCount = 1500;
    double idPercentile = 95.0;
    std::string outputPath;
    bool fetchIfMissing = true;
    LogLevel logLevel = LogLevel::ERROR;
    ulong progressReports = 0;
  };

  //===================================================================================================================//

  //---- File / image helpers ---------------------------------------------------//

  /// Return true if *p* has a recognised image extension (.jpg, .jpeg, .png, .bmp).
  inline bool isImagePath(const std::filesystem::path& p)
  {
    if (!p.has_extension())
      return false;

    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
  }

  /// Recursively check whether *root* contains any recognised image files.
  bool dirHasImages(const std::filesystem::path& root);

  /// Recursively collect absolute paths of all recognised images under *rootDir*, sorted.
  std::vector<std::string> gatherImages(const std::string& rootDir);

  /// Deterministic random subsample of *all* (seeded, for reproducibility).
  std::vector<std::string> sampleImages(const std::vector<std::string>& all, std::size_t count, uint32_t seed);

  //===================================================================================================================//

  //---- Free-energy / statistics helpers ---------------------------------------//

  /// log-sum-exp-stable free-energy: E = −log Σ exp(zᵢ).
  float computeFreeEnergy(const std::vector<float>& logits);

  /// Linear-interpolation percentile on a *sorted* vector.
  float computePercentile(const std::vector<float>& sorted, double p);

  /// Round a double to the requested number of decimal places (for cleaner JSON).
  double roundTo(double v, int places);

  //===================================================================================================================//

  //---- OOD dataset providers --------------------------------------------------//

  /// Run a command via QProcess (no shell injection risk).
  /// Throws std::runtime_error on failure (timeout, non-zero exit, not found).
  void shellRun(const QString& program, const QStringList& arguments, const std::string& description);

  /// Download/generate DTD + Places365-val_256 + synthetic OOD images if missing.
  /// @param oodDir        Target OOD directory.
  /// @param logLevel      Log level for conditional messages.
  /// @param logCallback   Observer callback: (message, isError).
  void ensureOODDataset(const std::string& oodDir, LogLevel logLevel,
                        std::function<void(const std::string&, bool)> logCallback);

  /// Download and extract the DTD dataset into *destDir*.
  void fetchDTD(const std::string& destDir);

  /// Download and extract Places365 val_256 into *destDir*.
  void fetchPlaces365Val(const std::string& destDir);

  /// Generate synthetic OOD images (solid colours, splits, checkerboard, noise) into *destDir*.
  void generateSynthetic(const std::string& destDir);

  //===================================================================================================================//

  //---- runPredictImpl (HEADER-ONLY TEMPLATE) ----------------------------------//

  /**
   * Unified streaming predict for ANN / CNN.
   *
   * @tparam InputsT   CNN::Inputs<float> or ANN::Inputs<float>
   * @tparam CoreT     CNN::Core<float>   or ANN::Core<float>
   * @tparam WrapFn    callable
   *                   (std::vector<float>&& flat) → element of InputsT
   *
   * Loads images in parallel via QtConcurrent::blockingMap, wraps each flat
   * pixel vector with *wrapInput*, feeds it through *core.predict()* batch
   * by batch, and returns the per-sample logits (pre-activation z values).
   */
  template <typename InputsT, typename CoreT, typename WrapFn>
  std::vector<std::vector<float>>
  runPredictImpl(CoreT& core, const std::vector<std::string>& imagePaths, const std::string& progressLabel, int targetC,
                 int targetH, int targetW, ulong progressReports, LogLevel logLevel, WrapFn wrapInput)
  {
    std::vector<std::vector<float>> result;
    ulong total = imagePaths.size();
    std::atomic<ulong> loadedCount{0};

    auto provider = [&](ulong batchSize, ulong batchIndex) -> InputsT {
      ulong start = batchIndex * batchSize;
      ulong end = std::min(start + batchSize, total);

      if (start >= end)
        return {};

      ulong batchN = end - start;
      std::vector<std::vector<float>> flats(batchN);
      std::vector<char> ok(batchN, 0);

      QVector<int> indices(static_cast<int>(batchN));

      for (int k = 0; k < static_cast<int>(batchN); k++)
        indices[k] = k;

      QtConcurrent::blockingMap(indices, [&](int k) {
        try {
          flats[k] = ImageLoader::loadImage(imagePaths[start + k], targetC, targetH, targetW);
          ok[k] = 1;
        } catch (const std::exception& e) {
          if (logLevel >= LogLevel::WARNING)
            std::cerr << "\n[warn] Skipping " << imagePaths[start + k] << ": " << e.what() << "\n";
        }

        ulong done = ++loadedCount;

        if (logLevel > LogLevel::QUIET)
          printLoadingProgress(std::string("Loading ") + progressLabel, done, total, progressReports);
      });

      InputsT inputs;
      inputs.reserve(batchN);

      for (ulong k = 0; k < batchN; k++) {
        if (!ok[k])
          continue;

        inputs.push_back(wrapInput(std::move(flats[k])));
      }

      return inputs;
    };

    if (logLevel > LogLevel::QUIET) {
      printLoadingProgress(std::string("Loading ") + progressLabel, 0, total, progressReports);
      core.setProgressCallback([progressReports, &progressLabel](ulong current, ulong totalCb) {
        printLoadingProgress(progressLabel, current, totalCb, progressReports);
      });
    }

    auto predicts = core.predict(total, provider);
    result.reserve(predicts.size());

    for (auto& p : predicts)
      result.push_back(std::move(p.logits));

    return result;
  }

} // namespace NN_CLI

//===================================================================================================================//

#endif // NN_CLI_CALIBRATEUTILS_HPP
