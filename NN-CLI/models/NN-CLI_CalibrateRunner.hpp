#ifndef NN_CLI_CALIBRATERUNNER_HPP
#define NN_CLI_CALIBRATERUNNER_HPP

#include "NN-CLI_AugmentationConfig.hpp"
#include "NN-CLI_CalibrateUtils.hpp"
#include "NN-CLI_IOConfig.hpp"
#include "NN-CLI_LogLevel.hpp"
#include "NN-CLI_NetworkType.hpp"
#include "NN-CLI_RunnerObserver.hpp"

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
 * The runner consumes a Core that the parent App created in PREDICT
 * mode — calibrate is a CLI-level mode, not a model-level one.
 */
  class CalibrateRunner : public IRunnerObserver
  {
    public:
      //-- Ctors / Dtors --//

      CalibrateRunner(const QCommandLineParser& parser, LogLevel logLevel, NetworkType networkType,
                      const IOConfig& ioConfig, const AugmentationConfig& augConfig,
                      std::unique_ptr<ANN::Core<float>>& annCore, const ANN::CoreConfig<float>& annCoreConfig,
                      std::unique_ptr<CNN::Core<float>>& cnnCore, const CNN::CoreConfig<float>& cnnCoreConfig);

      //-- Observer management --//
      void addObserver(IRunnerObserver* observer);
      void removeObserver(IRunnerObserver* observer);

      //-- IRunnerObserver overrides (no-op — used when CalibrateRunner acts as observer on itself) --//
      void onBatchProgress(int batchIdx, int totalBatches, float currentLoss, float samplesPerSec, float etaSeconds,
                           const std::vector<float>& fractions) override;
      void onEpochCompleted(int epochIdx, int totalEpochs, float epochLoss, bool hasValLoss, float valLoss,
                            const std::string& summary) override;
      void onTrainingFinished(bool success, const std::string& finalSummary) override;
      void onModelInfoUpdated(const std::string& property, const std::string& value) override;
      void onLogMessage(const std::string& message, bool isError) override;
      void onTimingUpdated(const std::string& metric, float value) override;

      //-- Lifecycle --//

      int run();

    private:
      //-- Observer notifications --//
      void notifyLogMessage(const std::string& message, bool isError);
      void notifyTrainingFinished(bool success, const std::string& finalSummary);

      //-- Members --//
      const QCommandLineParser& parser;
      LogLevel logLevel;
      NetworkType networkType;
      IOConfig ioConfig;
      AugmentationConfig augConfig;

      // Cores are owned by the parent App; we hold references.
      std::unique_ptr<ANN::Core<float>>& annCore;
      ANN::CoreConfig<float> annCoreConfig;
      std::unique_ptr<CNN::Core<float>>& cnnCore;
      CNN::CoreConfig<float> cnnCoreConfig;

      //-- Observer list --//
      std::vector<IRunnerObserver*> observers;
  };
} // namespace NN_CLI

//===================================================================================================================//

#endif // NN_CLI_CALIBRATERUNNER_HPP
