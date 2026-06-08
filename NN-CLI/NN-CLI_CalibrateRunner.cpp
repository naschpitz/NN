#include "NN-CLI_CalibrateRunner.hpp"

#include "NN-CLI_ImageLoader.hpp"
#include "NN-CLI_ProgressBar.hpp"

#include <ANN_Utils.hpp>

#include <json.hpp>

#include <QFile>
#include <QProcess>
#include <QThreadPool>
#include <QVector>
#include <QtConcurrent>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using namespace NN_CLI;

//===================================================================================================================//
//-- Helpers (file-local) --//
//===================================================================================================================//

namespace
{
  // Lower-cased extensions we accept as images.
  bool isImagePath(const fs::path& p)
  {
    if (!p.has_extension())
      return false;

    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
  }

  // log-sum-exp-stable free-energy: E = −log Σ exp(zᵢ).
  float freeEnergy(const std::vector<float>& logits)
  {
    if (logits.empty())
      return std::numeric_limits<float>::infinity();

    float m = *std::max_element(logits.begin(), logits.end());
    float sumExp = 0.0f;

    for (float z : logits)
      sumExp += std::exp(z - m);

    return -(m + std::log(sumExp));
  }

  // Linear-interpolation percentile on a *sorted* vector.
  float percentile(const std::vector<float>& sorted, double p)
  {
    if (sorted.empty())
      return std::numeric_limits<float>::quiet_NaN();

    if (sorted.size() == 1)
      return sorted.front();

    double idx = (sorted.size() - 1) * p / 100.0;
    std::size_t lo = static_cast<std::size_t>(idx);
    std::size_t hi = std::min(lo + 1, sorted.size() - 1);
    double frac = idx - static_cast<double>(lo);
    return static_cast<float>(sorted[lo] + (sorted[hi] - sorted[lo]) * frac);
  }

  bool dirHasImages(const fs::path& root)
  {
    if (!fs::exists(root) || !fs::is_directory(root))
      return false;

    for (const auto& e : fs::recursive_directory_iterator(root)) {
      if (e.is_regular_file() && isImagePath(e.path()))
        return true;
    }

    return false;
  }

  // Run a command via QProcess (no shell injection risk).
  // We shell out to `curl` / `tar` because pulling in libcurl/libarchive
  // for a one-time calibration setup isn't worth the link-time cost.
  void shellRun(const QString& program, const QStringList& arguments, const std::string& description)
  {
    QProcess process;
    process.start(program, arguments);

    if (!process.waitForFinished(600000)) { // 10-minute timeout
      process.kill();

      if (process.error() == QProcess::FailedToStart) {
        throw std::runtime_error(description + " failed to start: " + program.toStdString());
      }

      QString cmd = program + " " + arguments.join(" ");
      throw std::runtime_error(description + " timed out: " + cmd.toStdString());
    }

    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
      QString cmd = program + " " + arguments.join(" ");
      QString err = QString::fromUtf8(process.readAllStandardError());

      throw std::runtime_error(description + " failed (exit " + std::to_string(process.exitCode()) +
                               "): " + cmd.toStdString() + "\n" + err.toStdString());
    }
  }

  // Round a double to the requested number of decimal places (for cleaner JSON).
  double roundTo(double v, int places)
  {
    double m = std::pow(10.0, places);
    return std::round(v * m) / m;
  }

  //===================================================================================================================//
  //-- Template helpers (file-local) --//
  //===================================================================================================================//

  // Unified streaming predict for /CNN – the only difference is how
  // a decoded flat pixel vector is wrapped into the library's input type.
  // InputsT:       CNN::Inputs<float> or ANN::Inputs<float>
  // CoreT:         CNN::Core<float>   or ANN::Core<float>
  // WrapFn:        callable (std::vector<float>&&) → element of InputsT
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
          ProgressBar::printLoadingProgress(std::string("Loading ") + progressLabel, done, total, progressReports);
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
      ProgressBar::printLoadingProgress(std::string("Loading ") + progressLabel, 0, total, progressReports);
      core.setProgressCallback([progressReports, &progressLabel](ulong current, ulong totalCb) {
        ProgressBar::printLoadingProgress(progressLabel, current, totalCb, progressReports);
      });
    }

    auto predicts = core.predict(total, provider);
    result.reserve(predicts.size());

    for (auto& p : predicts)
      result.push_back(std::move(p.logits));

    return result;
  }
} // namespace

//===================================================================================================================//
//-- Constructor --//
//===================================================================================================================//

CalibrateRunner::CalibrateRunner(const QCommandLineParser& parser, LogLevel logLevel, NetworkType networkType,
                                 const IOConfig& ioConfig, const AugmentationConfig& augConfig,
                                 std::unique_ptr<ANN::Core<float>>& annCore,
                                 const ANN::CoreConfig<float>& annCoreConfig,
                                 std::unique_ptr<CNN::Core<float>>& cnnCore,
                                 const CNN::CoreConfig<float>& cnnCoreConfig)
  : parser(parser),
    logLevel(logLevel),
    networkType(networkType),
    ioConfig(ioConfig),
    augConfig(augConfig),
    annCore(annCore),
    annCoreConfig(annCoreConfig),
    cnnCore(cnnCore),
    cnnCoreConfig(cnnCoreConfig)
{
}

//===================================================================================================================//
//-- run --//
//===================================================================================================================//

int CalibrateRunner::run()
{
  // ---- args --------------------------------------------------------------
  if (!this->parser.isSet("id-images")) {
    std::cerr << "Error: --id-images <dir> is required for calibrate mode.\n";
    return 1;
  }

  std::string idDir = this->parser.value("id-images").toStdString();

  std::string oodDir = this->parser.isSet("ood-dir") ? this->parser.value("ood-dir").toStdString()
                                                     : (fs::current_path() / "extern-datasets" / "ood").string();

  std::size_t idCount = 500;

  if (this->parser.isSet("id-sample-count")) {
    bool ok = false;
    idCount = this->parser.value("id-sample-count").toULongLong(&ok);

    if (!ok)
      throw std::runtime_error("Error: --id-sample-count must be a positive integer");
  }

  std::size_t oodCount = 1500;

  if (this->parser.isSet("ood-sample-count")) {
    bool ok = false;
    oodCount = this->parser.value("ood-sample-count").toULongLong(&ok);

    if (!ok)
      throw std::runtime_error("Error: --ood-sample-count must be a positive integer");
  }

  double idPercentile = 95.0;

  if (this->parser.isSet("id-percentile")) {
    bool ok = false;
    idPercentile = this->parser.value("id-percentile").toDouble(&ok);

    if (!ok)
      throw std::runtime_error("Error: --id-percentile must be a number");

    if (idPercentile < 0.0 || idPercentile > 100.0)
      throw std::runtime_error("Error: --id-percentile must be between 0 and 100");
  }

  bool fetchIfMissing = !this->parser.isSet("no-fetch");

  // Default output: threshold.json next to --config.
  std::string outputPath;

  if (this->parser.isSet("output")) {
    outputPath = this->parser.value("output").toStdString();
  } else {
    fs::path configPath = this->parser.value("config").toStdString();
    outputPath = (configPath.parent_path() / "threshold.json").string();
  }

  if (this->logLevel > LogLevel::QUIET) {
    std::cout << "Calibrate mode\n";
    std::cout << "  Model:           " << this->parser.value("config").toStdString() << "\n";
    std::cout << "  ID images:       " << idDir << "  (sample " << idCount << ")\n";
    std::cout << "  OOD dir:         " << oodDir << "  (sample " << oodCount << ")\n";
    std::cout << "  ID percentile:   " << idPercentile << "\n";
    std::cout << "  Output:          " << outputPath << "\n";
    std::cout << "\n";
  }

  // ---- fetch OOD if needed ----------------------------------------------
  if (fetchIfMissing && !dirHasImages(oodDir)) {
    if (this->logLevel > LogLevel::QUIET)
      std::cout << "OOD dir is empty — fetching DTD + Places365 + synthetic.\n";

    this->ensureOODDataset(oodDir);
  } else if (!dirHasImages(oodDir)) {
    std::cerr << "Error: --ood-dir " << oodDir << " has no images and --no-fetch was set.\n";
    return 1;
  }

  // ---- gather + sample --------------------------------------------------
  std::vector<std::string> idAll = gatherImages(idDir);
  std::vector<std::string> oodAll = gatherImages(oodDir);

  if (idAll.empty()) {
    std::cerr << "Error: no images found under --id-images " << idDir << "\n";
    return 1;
  }

  if (oodAll.empty()) {
    std::cerr << "Error: no images found under --ood-dir " << oodDir << "\n";
    return 1;
  }

  std::vector<std::string> idSample = sampleImages(idAll, idCount, /*seed=*/42);
  std::vector<std::string> oodSample = sampleImages(oodAll, oodCount, /*seed=*/42);

  if (this->logLevel > LogLevel::QUIET) {
    std::cout << "Sampled " << idSample.size() << " ID images (of " << idAll.size() << " available)\n";
    std::cout << "Sampled " << oodSample.size() << " OOD images (of " << oodAll.size() << " available)\n\n";
  }

  // ---- predict + free-energy --------------------------------------------
  auto t0 = std::chrono::system_clock::now();

  std::vector<std::vector<float>> idLogits = this->runPredict(idSample, "Predicting ID");
  std::vector<std::vector<float>> oodLogits = this->runPredict(oodSample, "Predicting OOD");

  std::vector<float> idEnergies, oodEnergies;
  idEnergies.reserve(idLogits.size());
  oodEnergies.reserve(oodLogits.size());

  for (const auto& l : idLogits)
    idEnergies.push_back(freeEnergy(l));

  for (const auto& l : oodLogits)
    oodEnergies.push_back(freeEnergy(l));

  std::sort(idEnergies.begin(), idEnergies.end());
  std::sort(oodEnergies.begin(), oodEnergies.end());

  // ---- write threshold.json ---------------------------------------------
  this->writeThresholdJson(outputPath, idEnergies, oodEnergies, idPercentile);

  auto t1 = std::chrono::system_clock::now();

  if (this->logLevel > LogLevel::QUIET) {
    std::chrono::duration<double> elapsed = t1 - t0;
    std::cout << "\nCalibration done in " << ANN::Utils<float>::formatDuration(elapsed.count()) << "\n";
    std::cout << "Threshold written to: " << outputPath << "\n";
  }

  return 0;
}

//===================================================================================================================//
//-- ensureOODDataset --//
//===================================================================================================================//

void CalibrateRunner::ensureOODDataset(const std::string& oodDir)
{
  fs::create_directories(oodDir);

  fs::path dtdDir = fs::path(oodDir) / "dtd";
  fs::path placesDir = fs::path(oodDir) / "places365-val";
  fs::path synDir = fs::path(oodDir) / "synthetic";

  if (!dirHasImages(dtdDir))
    this->fetchDTD(dtdDir.string());
  else if (this->logLevel > LogLevel::QUIET)
    std::cout << "[skip] DTD already extracted at " << dtdDir.string() << "\n";

  if (!dirHasImages(placesDir))
    this->fetchPlaces365Val(placesDir.string());
  else if (this->logLevel > LogLevel::QUIET)
    std::cout << "[skip] Places365 val_256 already extracted at " << placesDir.string() << "\n";

  if (!dirHasImages(synDir))
    this->generateSynthetic(synDir.string());
  else if (this->logLevel > LogLevel::QUIET)
    std::cout << "[skip] Synthetic OOD already at " << synDir.string() << "\n";
}

//===================================================================================================================//
//-- fetchDTD --//
//===================================================================================================================//

void CalibrateRunner::fetchDTD(const std::string& destDir)
{
  fs::create_directories(destDir);
  fs::path tgz = fs::path(destDir).parent_path() / "dtd-r1.0.1.tar.gz";

  if (this->logLevel > LogLevel::QUIET)
    std::cout << "[fetch] DTD (~600 MB) → " << destDir << "\n";

  shellRun("curl",
           {"-L", "--fail", "-o", QString::fromStdString(tgz.string()),
            "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"},
           "DTD download");
  shellRun(
    "tar",
    {"-xzf", QString::fromStdString(tgz.string()), "-C", QString::fromStdString(destDir), "--strip-components=1"},
    "DTD extraction");
  fs::remove(tgz);
}

//===================================================================================================================//
//-- fetchPlaces365Val --//
//===================================================================================================================//

void CalibrateRunner::fetchPlaces365Val(const std::string& destDir)
{
  fs::create_directories(destDir);
  fs::path tarPath = fs::path(destDir).parent_path() / "val_256.tar";

  if (this->logLevel > LogLevel::QUIET)
    std::cout << "[fetch] Places365 val_256 (~500 MB) → " << destDir << "\n";

  shellRun("curl",
           {"-L", "--fail", "-o", QString::fromStdString(tarPath.string()),
            "http://data.csail.mit.edu/places/places365/val_256.tar"},
           "Places365 download");
  shellRun("tar", {"-xf", QString::fromStdString(tarPath.string()), "-C", QString::fromStdString(destDir)},
           "Places365 extraction");
  fs::remove(tarPath);
}

//===================================================================================================================//
//-- generateSynthetic --//
//===================================================================================================================//

void CalibrateRunner::generateSynthetic(const std::string& destDir)
{
  fs::create_directories(destDir);

  if (this->logLevel > LogLevel::QUIET)
    std::cout << "[gen]   Synthetic OOD → " << destDir << "\n";

  // 3-channel 256×256 (NN-CLI's ImageLoader will resize to the model's inputShape).
  const int C = 3, H = 256, W = 256;

  auto saveSolid = [&](unsigned char r, unsigned char g, unsigned char b, const std::string& name) {
    std::vector<float> data(static_cast<std::size_t>(C) * H * W, 0.0f);

    for (int i = 0; i < H * W; i++) {
      data[0 * H * W + i] = r / 255.0f;
      data[1 * H * W + i] = g / 255.0f;
      data[2 * H * W + i] = b / 255.0f;
    }

    ImageLoader::saveImage((fs::path(destDir) / ("solid_" + name + ".jpg")).string(), data, C, H, W);
  };

  saveSolid(0, 0, 0, "black");
  saveSolid(255, 255, 255, "white");
  saveSolid(128, 128, 128, "gray");
  saveSolid(255, 0, 0, "red");
  saveSolid(0, 255, 0, "green");
  saveSolid(0, 0, 255, "blue");
  saveSolid(255, 255, 0, "yellow");
  saveSolid(255, 0, 255, "magenta");
  saveSolid(0, 255, 255, "cyan");
  saveSolid(230, 200, 170, "skin_like");

  // Half-split black/white.
  {
    std::vector<float> data(static_cast<std::size_t>(C) * H * W, 0.0f);

    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        float v = (w < W / 2) ? 0.0f : 1.0f;

        for (int c = 0; c < C; c++)
          data[c * H * W + h * W + w] = v;
      }
    }

    ImageLoader::saveImage((fs::path(destDir) / "split_black_white.jpg").string(), data, C, H, W);
  }

  // Checkerboard.
  {
    std::vector<float> data(static_cast<std::size_t>(C) * H * W, 0.0f);
    const int n = 16;
    const int cell = H / n;

    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        int gi = h / cell;
        int gj = w / cell;
        float v = ((gi + gj) % 2 == 0) ? 1.0f : 0.0f;

        for (int c = 0; c < C; c++)
          data[c * H * W + h * W + w] = v;
      }
    }

    ImageLoader::saveImage((fs::path(destDir) / "checker_bw.jpg").string(), data, C, H, W);
  }

  // Random noise (uniform).
  {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);

    for (int idx = 0; idx < 20; idx++) {
      std::vector<float> data(static_cast<std::size_t>(C) * H * W, 0.0f);

      for (auto& v : data)
        v = u(rng);

      char buf[64];
      std::snprintf(buf, sizeof(buf), "noise_uniform_%03d.jpg", idx);
      ImageLoader::saveImage((fs::path(destDir) / buf).string(), data, C, H, W);
    }
  }

  // Random low-frequency (out-of-focus blur).
  {
    std::mt19937 rng(43);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    const int LOW = 16; // Render LOW×LOW noise, nearest-neighbor upsample to H×W → blocky low-freq texture.

    for (int idx = 0; idx < 20; idx++) {
      std::vector<float> small(static_cast<std::size_t>(C) * LOW * LOW, 0.0f);

      for (auto& v : small)
        v = u(rng);

      std::vector<float> data(static_cast<std::size_t>(C) * H * W, 0.0f);

      for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
          for (int w = 0; w < W; w++) {
            int hs = (h * LOW) / H;
            int ws = (w * LOW) / W;
            data[c * H * W + h * W + w] = small[c * LOW * LOW + hs * LOW + ws];
          }
        }
      }

      char buf[64];
      std::snprintf(buf, sizeof(buf), "noise_blocky_%03d.jpg", idx);
      ImageLoader::saveImage((fs::path(destDir) / buf).string(), data, C, H, W);
    }
  }
}

//===================================================================================================================//
//-- gatherImages --//
//===================================================================================================================//

std::vector<std::string> CalibrateRunner::gatherImages(const std::string& rootDir)
{
  std::vector<std::string> result;

  if (!fs::exists(rootDir))
    return result;

  for (const auto& entry : fs::recursive_directory_iterator(rootDir)) {
    if (entry.is_regular_file() && isImagePath(entry.path()))
      result.push_back(fs::absolute(entry.path()).string());
  }

  std::sort(result.begin(), result.end());
  return result;
}

//===================================================================================================================//
//-- sampleImages --//
//===================================================================================================================//

std::vector<std::string> CalibrateRunner::sampleImages(const std::vector<std::string>& all, std::size_t count,
                                                       uint32_t seed)
{
  if (count >= all.size())
    return all;

  std::vector<std::string> shuffled = all;
  std::mt19937 rng(seed);
  std::shuffle(shuffled.begin(), shuffled.end(), rng);
  shuffled.resize(count);
  return shuffled;
}

//===================================================================================================================//
//-- runPredict --//
//===================================================================================================================//

std::vector<std::vector<float>> CalibrateRunner::runPredict(const std::vector<std::string>& imagePaths,
                                                            const std::string& progressLabel)
{
  ulong total = imagePaths.size();
  ulong progressReports = (this->logLevel > LogLevel::QUIET) ? this->ioConfig.progressReports : 0;

  // Pick target dimensions (CNN uses 3D inputShape; 's image path uses ioConfig).
  int targetC = (this->networkType == NetworkType::CNN) ? static_cast<int>(this->cnnCoreConfig.inputShape.c)
                                                        : static_cast<int>(this->ioConfig.inputC);
  int targetH = (this->networkType == NetworkType::CNN) ? static_cast<int>(this->cnnCoreConfig.inputShape.h)
                                                        : static_cast<int>(this->ioConfig.inputH);
  int targetW = (this->networkType == NetworkType::CNN) ? static_cast<int>(this->cnnCoreConfig.inputShape.w)
                                                        : static_cast<int>(this->ioConfig.inputW);

  std::vector<std::vector<float>> result;

  if (this->networkType == NetworkType::CNN) {
    auto shape = this->cnnCoreConfig.inputShape;

    auto wrapInput = [shape](std::vector<float>&& flat) {
      CNN::Input<float> input(shape);
      input.data = std::move(flat);
      return input;
    };

    result = runPredictImpl<CNN::Inputs<float>>(*this->cnnCore, imagePaths, progressLabel, targetC, targetH, targetW,
                                                progressReports, this->logLevel, wrapInput);
  } else {
    auto wrapInput = [](std::vector<float>&& flat) {
      return std::move(flat);
    };

    result = runPredictImpl<ANN::Inputs<float>>(*this->annCore, imagePaths, progressLabel, targetC, targetH, targetW,
                                                progressReports, this->logLevel, wrapInput);
  }

  if (this->logLevel > LogLevel::QUIET)
    std::cout << "\n";

  return result;
}

//===================================================================================================================//
//-- writeThresholdJson --//
//===================================================================================================================//

void CalibrateRunner::writeThresholdJson(const std::string& outputPath, const std::vector<float>& idSorted,
                                         const std::vector<float>& oodSorted, double idPercentile)
{
  auto stats = [](const std::vector<float>& sorted, const std::vector<double>& ps) {
    nlohmann::ordered_json out;
    out["n"] = sorted.size();
    out["min"] = roundTo(sorted.front(), 4);
    out["max"] = roundTo(sorted.back(), 4);

    double mean = 0.0;

    for (float v : sorted)
      mean += v;

    mean /= sorted.size();

    out["mean"] = roundTo(mean, 4);

    for (double p : ps) {
      char key[16];
      std::snprintf(key, sizeof(key), "p%g", p);
      out[key] = roundTo(percentile(sorted, p), 4);
    }

    return out;
  };

  float threshold = percentile(idSorted, idPercentile);

  std::size_t idAccepted = 0;

  for (float e : idSorted) {
    if (e <= threshold)
      idAccepted++;
  }

  std::size_t oodRejected = 0;

  for (float e : oodSorted) {
    if (e > threshold)
      oodRejected++;
  }

  nlohmann::ordered_json doc;
  doc["freeEnergyThreshold"] = roundTo(threshold, 4);
  doc["idPercentileUsed"] = idPercentile;
  doc["rule"] = "predicted_ood = (free_energy > freeEnergyThreshold)";
  doc["idStats"] = stats(idSorted, {1, 5, 50, 90, 95, 99});
  doc["oodStats"] = stats(oodSorted, {1, 5, 50, 95, 99});

  nlohmann::ordered_json conf;
  conf["idAccepted"] = idAccepted;
  conf["idRejected"] = idSorted.size() - idAccepted;
  conf["oodAccepted"] = oodSorted.size() - oodRejected;
  conf["oodRejected"] = oodRejected;
  conf["idAcceptanceRate"] = roundTo(static_cast<double>(idAccepted) / idSorted.size(), 4);
  conf["oodRejectionRate"] = roundTo(static_cast<double>(oodRejected) / oodSorted.size(), 4);
  doc["confusion"] = conf;

  std::ofstream f(outputPath);

  if (!f)
    throw std::runtime_error("Cannot write " + outputPath);

  f << doc.dump(2) << "\n";

  if (this->logLevel > LogLevel::QUIET)
    std::cout << doc.dump(2) << "\n";
}
