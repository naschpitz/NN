#ifndef NN_CLI_CNNRUNNER_HPP
#define NN_CLI_CNNRUNNER_HPP

#include "NN-CLI_AugmentationConfig.hpp"
#include "NN-CLI_DataLoader.hpp"
#include "NN-CLI_IOConfig.hpp"
#include "NN-CLI_LogLevel.hpp"
#include "NN-CLI_ModelSerializer.hpp"
#include "NN-CLI_TerminalUI.hpp"
#include "NN-CLI_TrainingProfiler.hpp"
#include "NN-CLI_TrainingTui.hpp"
#include "NN-CLI_ProgressBar.hpp"
#include "NN-CLI_Utils.hpp"

#include <CNN_Core.hpp>
#include "Common/Common_TrainingMonitor.hpp"

#include <QCommandLineParser>

#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

//===================================================================================================================//

namespace NN_CLI
{

  class CNNRunner
  {
    public:
      CNNRunner(const QCommandLineParser& parser, LogLevel logLevel, IOConfig& ioConfig, AugmentationConfig& augConfig,
                std::unique_ptr<CNN::Core<float>>& core, CNN::CoreConfig<float>& coreConfig);

      int train();
      int test();
      int predict();

    private:
      //-- Sample loading --//
      std::pair<CNN::Samples<float>, bool> loadSamplesFromOptions(const std::string& modeName, QString& inputFilePath);

      //-- Training helpers --//
      void setupTrainingCallback(const QString& inputFilePath,
                                 std::shared_ptr<CNN::Core<float>> validationCore = nullptr,
                                 std::shared_ptr<Common::TrainingMonitor<float>> trainingMonitor = nullptr,
                                 const DataLoader<CNN::Sample<float>>* validationDataLoader = nullptr,
                                 const std::vector<ulong>* validationIndices = nullptr);
      int finishTraining(const QString& inputFilePath);
      ValidationMetadata buildValidationMetadata() const;

      //-- References to shared state (owned by Runner) --//
      const QCommandLineParser& parser;
      LogLevel logLevel;
      IOConfig& ioConfig;
      AugmentationConfig& augConfig;
      std::unique_ptr<CNN::Core<float>>& core;
      CNN::CoreConfig<float>& coreConfig;

      //-- Validation state --//
      ValidationState validationState;

      //-- Per-phase timing profiler (fed by CNN's timing callback) --//
      TrainingProfiler profiler;

      //-- ncurses terminal UI (only active during training) --//
      std::shared_ptr<TerminalUI> tui;

      //-- Loading-bar wiring shared with the  runner --//
      TrainingTui trainingTui;
      //-- Callback State --//
      ulong lastCallbackEpoch = 0;
      float lastEpochLoss = 0.0f;
      bool lastIsBest = false;
      bool lastHadValLoss = false;
      float lastValLoss = 0.0f;
      bool cacheIsSet = false;
      std::mutex epochTransitionMutex;
      std::unique_ptr<ProgressBar> progressBar;

      //-- Validation objects (stored during train() for finishTraining()) --//
      std::shared_ptr<CNN::Core<float>> validationCore;
      std::shared_ptr<CNN::SampleProvider<float>> validationProviderPtr;
      std::shared_ptr<std::vector<ulong>> validationIndices;
      std::shared_ptr<Common::TrainingMonitor<float>> trainingMonitor;
  };

} // namespace NN_CLI

#endif // NN_CLI_CNNRUNNER_HPP
