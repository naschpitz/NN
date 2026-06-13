#include "NN-CLI_CalibrateUtils.hpp"

#include "NN-CLI_ImageLoader.hpp"
#include "NN-CLI_Utils.hpp"

#include "Common/Common_Utils.hpp"

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

//===================================================================================================================//

namespace NN_CLI
{

  //===================================================================================================================//
  //-- dirHasImages --//
  //===================================================================================================================//

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

  //===================================================================================================================//

  //===================================================================================================================//
  //-- gatherImages --//
  //===================================================================================================================//

  std::vector<std::string> gatherImages(const std::string& rootDir)
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

  //===================================================================================================================//
  //-- sampleImages --//
  //===================================================================================================================//

  std::vector<std::string> sampleImages(const std::vector<std::string>& all, std::size_t count, uint32_t seed)
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

  //===================================================================================================================//
  //-- computeFreeEnergy --//
  //===================================================================================================================//

  float computeFreeEnergy(const std::vector<float>& logits)
  {
    if (logits.empty())
      return std::numeric_limits<float>::infinity();

    float m = *std::max_element(logits.begin(), logits.end());
    float sumExp = 0.0f;

    for (float z : logits)
      sumExp += std::exp(z - m);

    return -(m + std::log(sumExp));
  }

  //===================================================================================================================//

  //===================================================================================================================//
  //-- computePercentile --//
  //===================================================================================================================//

  float computePercentile(const std::vector<float>& sorted, double p)
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

  //===================================================================================================================//

  //===================================================================================================================//
  //-- roundTo --//
  //===================================================================================================================//

  double roundTo(double v, int places)
  {
    double m = std::pow(10.0, places);
    return std::round(v * m) / m;
  }

  //===================================================================================================================//

  //===================================================================================================================//
  //-- shellRun --//
  //===================================================================================================================//

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

  //===================================================================================================================//

  //===================================================================================================================//
  //-- ensureOODDataset --//
  //===================================================================================================================//

  void ensureOODDataset(const std::string& oodDir, LogLevel logLevel,
                        std::function<void(const std::string&, bool)> logCallback)
  {
    fs::create_directories(oodDir);

    fs::path dtdDir = fs::path(oodDir) / "dtd";
    fs::path placesDir = fs::path(oodDir) / "places365-val";
    fs::path synDir = fs::path(oodDir) / "synthetic";

    if (!dirHasImages(dtdDir))
      fetchDTD(dtdDir.string());
    else if (logLevel > LogLevel::QUIET) {
      std::string msg = "[skip] DTD already extracted at " + dtdDir.string() + "\n";
      std::cout << msg;
      if (logCallback)
        logCallback(msg, false);
    }

    if (!dirHasImages(placesDir))
      fetchPlaces365Val(placesDir.string());
    else if (logLevel > LogLevel::QUIET) {
      std::string msg = "[skip] Places365 val_256 already extracted at " + placesDir.string() + "\n";
      std::cout << msg;
      if (logCallback)
        logCallback(msg, false);
    }

    if (!dirHasImages(synDir))
      generateSynthetic(synDir.string());
    else if (logLevel > LogLevel::QUIET) {
      std::string msg = "[skip] Synthetic OOD already at " + synDir.string() + "\n";
      std::cout << msg;
      if (logCallback)
        logCallback(msg, false);
    }
  }

  //===================================================================================================================//

  //===================================================================================================================//
  //-- fetchDTD --//
  //===================================================================================================================//

  void fetchDTD(const std::string& destDir)
  {
    fs::create_directories(destDir);
    fs::path tgz = fs::path(destDir).parent_path() / "dtd-r1.0.1.tar.gz";

    std::string msg = "[fetch] DTD (~600 MB) → " + destDir + "\n";
    std::cout << msg;

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

  //===================================================================================================================//
  //-- fetchPlaces365Val --//
  //===================================================================================================================//

  void fetchPlaces365Val(const std::string& destDir)
  {
    fs::create_directories(destDir);
    fs::path tarPath = fs::path(destDir).parent_path() / "val_256.tar";

    std::string msg = "[fetch] Places365 val_256 (~500 MB) → " + destDir + "\n";
    std::cout << msg;

    shellRun("curl",
             {"-L", "--fail", "-o", QString::fromStdString(tarPath.string()),
              "http://data.csail.mit.edu/places/places365/val_256.tar"},
             "Places365 download");
    shellRun("tar", {"-xf", QString::fromStdString(tarPath.string()), "-C", QString::fromStdString(destDir)},
             "Places365 extraction");
    fs::remove(tarPath);
  }

  //===================================================================================================================//

  //===================================================================================================================//
  //-- generateSynthetic --//
  //===================================================================================================================//

  void generateSynthetic(const std::string& destDir)
  {
    fs::create_directories(destDir);

    std::string msg = "[gen]   Synthetic OOD → " + destDir + "\n";
    std::cout << msg;

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

} // namespace NN_CLI

//===================================================================================================================//
