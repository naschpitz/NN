#ifndef NN_CLI_CALIBRATERUNNER_HPP
#define NN_CLI_CALIBRATERUNNER_HPP

#include "NN-CLI_AugmentationConfig.hpp"
#include "NN-CLI_IOConfig.hpp"
#include "NN-CLI_LogLevel.hpp"
#include "NN-CLI_NetworkType.hpp"

#include <ANN_Core.hpp>
#include <ANN_CoreConfig.hpp>
#include <CNN_Core.hpp>
#include <CNN_CoreConfig.hpp>

#include <QCommandLineParser>

#include <memory>
#include <string>
#include <vector>

//===================================================================================================================//

namespace NN_CLI
{
  /**
 * CalibrateRunner: picks a free-energy OOD threshold for a trained model.
 *
 * Free-energy:  E(x) = −log Σᵢ exp(zᵢ)
 *   Lower E  → input looks more in-distribution.
 *   Higher E → input is more likely out-of-distribution.
 *
 * Workflow:
 *   1. Recursively gather image paths from --id-images and --ood-dir.
 *   2. (Optional) If --ood-dir is empty and --fetch-if-missing is on,
 *      download DTD + Places365-val_256 + generate synthetic OOD.
 *   3. Random-sample N from each set (seeded for reproducibility).
 *   4. Run the model's predict pipeline on both samples to obtain logits.
 *   5. Compute E for every sample, sort, take the chosen ID percentile
 *      as the threshold.
 *   6. Emit threshold.json next to the model checkpoint.
 *
 * The runner consumes a Core that the parent Runner created in PREDICT
 * mode — calibrate is a CLI-level mode, not a model-level one.
 */
  class CalibrateRunner
  {
    public:
      CalibrateRunner(const QCommandLineParser& parser, LogLevel logLevel, NetworkType networkType,
                      const IOConfig& ioConfig, const AugmentationConfig& augConfig,
                      std::unique_ptr<ANN::Core<float>>& annCore, const ANN::CoreConfig<float>& annCoreConfig,
                      std::unique_ptr<CNN::Core<float>>& cnnCore, const CNN::CoreConfig<float>& cnnCoreConfig);

      int run();

    private:
      const QCommandLineParser& parser;
      LogLevel logLevel;
      NetworkType networkType;
      IOConfig ioConfig;
      AugmentationConfig augConfig;

      // Cores are owned by the parent Runner; we hold references.
      std::unique_ptr<ANN::Core<float>>& annCore;
      ANN::CoreConfig<float> annCoreConfig;
      std::unique_ptr<CNN::Core<float>>& cnnCore;
      CNN::CoreConfig<float> cnnCoreConfig;

      //-- Phases --//
      // Ensure --ood-dir contains DTD/Places365/synthetic; download/generate if missing.
      void ensureOODDataset(const std::string& oodDir);

      // Gather all image paths under root recursively (jpg/jpeg/png/bmp).
      static std::vector<std::string> gatherImages(const std::string& rootDir);

      // Random subsample (seeded), absolute paths.
      static std::vector<std::string> sampleImages(const std::vector<std::string>& all, std::size_t count,
                                                   uint32_t seed);

      // Run predict over the sampled images and return per-sample logits (length = numClasses).
      std::vector<std::vector<float>> runPredict(const std::vector<std::string>& imagePaths,
                                                 const std::string& progressLabel);

      //-- OOD dataset providers --//
      void fetchDTD(const std::string& destDir);
      void fetchPlaces365Val(const std::string& destDir);
      void generateSynthetic(const std::string& destDir);

      //-- Output --//
      void writeThresholdJson(const std::string& outputPath, const std::vector<float>& idEnergies,
                              const std::vector<float>& oodEnergies, double idPercentile);
  };
} // namespace NN_CLI

//===================================================================================================================//

#endif // NN_CLI_CALIBRATERUNNER_HPP
