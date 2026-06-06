#ifndef NN_CLI_CNNRUNNER_HPP
#define NN_CLI_CNNRUNNER_HPP

#include "NN-CLI_AugmentationConfig.hpp"
#include "NN-CLI_DataLoader.hpp"
#include "NN-CLI_IOConfig.hpp"
#include "NN-CLI_LogLevel.hpp"
#include "NN-CLI_ModelSerializer.hpp"
#include "NN-CLI_TerminalUI.hpp"
#include "NN-CLI_TrainingProfiler.hpp"

#include <CNN_Core.hpp>
#include <CNN_TrainingMonitor.hpp>

#include <QCommandLineParser>

#include <limits>
#include <memory>
#include <string>

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
                                 std::shared_ptr<CNN::TrainingMonitor<float>> trainingMonitor = nullptr,
                                 const DataLoader<CNN::Sample<float>>* validationDataLoader = nullptr,
                                 const std::vector<ulong>* validationIndices = nullptr);
      int finishTraining(const QString& inputFilePath);
      ValidationMetadata buildValidationMetadata() const;
      void regenerateConfigLines(ulong maxWidth);

      //-- Class weight computation --//
      static std::vector<float> computeClassWeightsFromOutputs(const std::vector<std::vector<float>>& outputs);

      //-- References to shared state (owned by Runner) --//
      const QCommandLineParser& parser;
      LogLevel logLevel;
      IOConfig& ioConfig;
      AugmentationConfig& augConfig;
      std::unique_ptr<CNN::Core<float>>& core;
      CNN::CoreConfig<float>& coreConfig;

      //-- Validation state --//
      struct ValidationState {
          bool enabled = false;
          ulong checkInterval = 1;
          ulong numValSamples = 0;
          float bestValLoss = std::numeric_limits<float>::max();
          ulong bestValEpoch = 0;
          float lastValLoss = 0.0f;
      };

      ValidationState validationState;

      //-- Per-phase timing profiler (fed by CNN's timing callback) --//
      TrainingProfiler profiler;

      //-- ncurses terminal UI (only active during training) --//
      std::shared_ptr<TerminalUI> tui;

      //-- Cached config for on-resize regeneration --//
      ulong cachedNumOrigTrainSamples_ = 0;
      ulong cachedNumTrainSamples_ = 0;
      ulong cachedNumValSamples_ = 0;
      float cachedValRatio_ = 0.0f;
      bool cachedValAuto_ = false;
      ulong cachedNumOutputClasses_ = 0;
      bool configLinesLoaded_ = false;
  };

} // namespace NN_CLI

#endif // NN_CLI_CNNRUNNER_HPP
