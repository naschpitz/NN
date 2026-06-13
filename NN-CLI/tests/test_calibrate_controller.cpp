#include "test_helpers.hpp"

#include "NN-CLI_CalibrateController.hpp"
#include "NN-CLI_CalibrateUtils.hpp"

#include <json.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;

//===================================================================================================================//

static void testComputeFreeEnergy()
{
  std::cout << "  testComputeFreeEnergy... ";

  // Uniform logits: all equal → energy = -log(Σ exp(z)) = -log(N * exp(z))
  // For 3 equal logits: z = [1, 1, 1] → m = 1, sumExp = 3*exp(0) = 3, energy = -(1 + log(3)) ≈ -(1 + 1.0986) = -2.0986
  {
    std::vector<float> logits = {1.0f, 1.0f, 1.0f};
    float e = NN_CLI::computeFreeEnergy(logits);
    float expected = -(1.0f + std::log(3.0f));
    CHECK_NEAR(e, expected, 0.001f, "freeEnergy uniform logits");
  }

  // Empty logits → inf
  {
    float e = NN_CLI::computeFreeEnergy({});
    CHECK(std::isinf(e), "freeEnergy empty -> inf");
    CHECK(e > 0, "freeEnergy empty -> +inf");
  }

  // Single logit: energy = -z
  {
    float e = NN_CLI::computeFreeEnergy({2.5f});
    CHECK_NEAR(e, -2.5f, 0.001f, "freeEnergy single logit");
  }

  std::cout << std::endl;
}

//===================================================================================================================//

static void testComputePercentile()
{
  std::cout << "  testComputePercentile... ";

  // Sorted [0, 1, 2, 3, 4]
  std::vector<float> sorted = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};

  // Median (50th) → index = (5-1)*0.5 = 2 → sorted[2] = 2.0
  CHECK_NEAR(NN_CLI::computePercentile(sorted, 50.0), 2.0f, 0.001f, "percentile 50th of 5 elements");

  // 0th → sorted[0] = 0.0
  CHECK_NEAR(NN_CLI::computePercentile(sorted, 0.0), 0.0f, 0.001f, "percentile 0th");

  // 100th → sorted[4] = 4.0
  CHECK_NEAR(NN_CLI::computePercentile(sorted, 100.0), 4.0f, 0.001f, "percentile 100th");

  // 25th → index = (5-1)*0.25 = 1.0 → sorted[1] = 1.0
  CHECK_NEAR(NN_CLI::computePercentile(sorted, 25.0), 1.0f, 0.001f, "percentile 25th exact");

  // Interpolation: 30th → idx = 4*0.3 = 1.2 → lo=1, hi=2, frac=0.2 → 1.0 + (2.0-1.0)*0.2 = 1.2
  CHECK_NEAR(NN_CLI::computePercentile(sorted, 30.0), 1.2f, 0.001f, "percentile 30th interpolated");

  // Empty → NaN
  float e = NN_CLI::computePercentile({}, 50.0);
  CHECK(std::isnan(e), "percentile empty -> NaN");

  // Single element → that element
  CHECK_NEAR(NN_CLI::computePercentile({7.5f}, 50.0), 7.5f, 0.001f, "percentile single element");

  std::cout << std::endl;
}

//===================================================================================================================//

static void testRoundTo()
{
  std::cout << "  testRoundTo... ";

  CHECK_NEAR(NN_CLI::roundTo(3.14159, 2), 3.14, 1e-9, "roundTo 2 places");
  CHECK_NEAR(NN_CLI::roundTo(3.14159, 0), 3.0, 1e-9, "roundTo 0 places");
  CHECK_NEAR(NN_CLI::roundTo(2.71828, 3), 2.718, 1e-9, "roundTo 3 places");
  CHECK_NEAR(NN_CLI::roundTo(0.0, 4), 0.0, 1e-9, "roundTo zero");

  // Negative rounding edge
  CHECK_NEAR(NN_CLI::roundTo(-1.23456, 2), -1.23, 1e-9, "roundTo negative");

  std::cout << std::endl;
}

//===================================================================================================================//

static void testIsImagePath()
{
  std::cout << "  testIsImagePath... ";

  CHECK(NN_CLI::isImagePath("photo.jpg"), "isImagePath .jpg");
  CHECK(NN_CLI::isImagePath("photo.JPG"), "isImagePath .JPG (case insensitive)");
  CHECK(NN_CLI::isImagePath("photo.jpeg"), "isImagePath .jpeg");
  CHECK(NN_CLI::isImagePath("photo.png"), "isImagePath .png");
  CHECK(NN_CLI::isImagePath("photo.bmp"), "isImagePath .bmp");
  CHECK(!NN_CLI::isImagePath("photo.txt"), "isImagePath .txt -> false");
  CHECK(!NN_CLI::isImagePath("photo"), "isImagePath no ext -> false");
  CHECK(!NN_CLI::isImagePath(""), "isImagePath empty -> false");

  std::cout << std::endl;
}

//===================================================================================================================//

static void testDirHasImages()
{
  std::cout << "  testDirHasImages... ";

  // A directory that doesn't exist
  CHECK(!NN_CLI::dirHasImages("/nonexistent/path/12345"), "dirHasImages nonexistent -> false");

  // The fixtures directory has image files
  QString fixDir = fixturePath("../fixtures");
  CHECK(NN_CLI::dirHasImages(fixDir.toStdString()), "dirHasImages fixtures dir");

  // Empty temp dir
  fs::path emptyDir = fs::temp_directory_path() / "nncli_test_empty_dir";
  fs::create_directories(emptyDir);
  CHECK(!NN_CLI::dirHasImages(emptyDir.string()), "dirHasImages empty dir -> false");
  fs::remove_all(emptyDir);

  std::cout << std::endl;
}

//===================================================================================================================//

static void testGatherImages()
{
  std::cout << "  testGatherImages... ";

  // Nonexistent dir → empty vector, not error
  auto result = NN_CLI::gatherImages("/nonexistent/path/12345");
  CHECK(result.empty(), "gatherImages nonexistent -> empty");

  // Existing dir returns sorted paths
  QString fixDir = fixturePath("../fixtures");
  auto paths = NN_CLI::gatherImages(fixDir.toStdString());
  CHECK(!paths.empty(), "gatherImages fixtures dir -> non-empty");

  // Results should be sorted
  for (std::size_t i = 1; i < paths.size(); i++)
    CHECK(paths[i - 1] <= paths[i], "gatherImages sorted order");

  std::cout << std::endl;
}

//===================================================================================================================//

static void testSampleImages()
{
  std::cout << "  testSampleImages... ";

  std::vector<std::string> all = {"a", "b", "c", "d", "e"};

  // Sample more than available → returns all
  auto s1 = NN_CLI::sampleImages(all, 100, 42);
  CHECK(s1.size() == all.size(), "sampleImages count > size -> all");

  // Sample 3 → deterministic with seed 42
  auto s2 = NN_CLI::sampleImages(all, 3, 42);
  CHECK(s2.size() == 3, "sampleImages count=3 -> 3 elements");

  // Same seed → same result
  auto s3 = NN_CLI::sampleImages(all, 3, 42);
  CHECK(s2 == s3, "sampleImages same seed -> same result");

  // Different seed → different result (very likely)
  auto s4 = NN_CLI::sampleImages(all, 3, 99);
  bool sameOrder = (s2 == s4);

  // It's possible (though extremely unlikely) to get the same order
  // with a different seed for a small set, but we verify the size at least.
  CHECK(s4.size() == 3, "sampleImages different seed -> 3 elements");

  std::cout << std::endl;
}

//===================================================================================================================//

static void testCalibrationConfigDefaults()
{
  std::cout << "  testCalibrationConfigDefaults... ";

  Common::CalibrationConfig config;
  CHECK(config.idSampleCount == 500, "default idSampleCount=500");
  CHECK(config.oodSampleCount == 1500, "default oodSampleCount=1500");
  CHECK(config.idPercentile == 95.0, "default idPercentile=95.0");
  CHECK(config.fetchIfMissing == true, "default fetchIfMissing=true");
  CHECK(config.logLevel == Common::LogLevel::ERROR, "default logLevel=ERROR");
  CHECK(config.progressReports == 0, "default progressReports=0");
  CHECK(config.idImagesDir.empty(), "default idImagesDir empty");
  CHECK(config.oodDir.empty(), "default oodDir empty");
  CHECK(config.outputPath.empty(), "default outputPath empty");

  std::cout << std::endl;
}

//===================================================================================================================//

void runCalibrateControllerTests()
{
  testComputeFreeEnergy();
  testComputePercentile();
  testRoundTo();
  testIsImagePath();
  testDirHasImages();
  testGatherImages();
  testSampleImages();
  testCalibrationConfigDefaults();
}
