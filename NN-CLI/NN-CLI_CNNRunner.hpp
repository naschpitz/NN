#ifndef NN_CLI_CNNRUNNER_HPP
#define NN_CLI_CNNRUNNER_HPP

#include "NN-CLI_Runner.hpp"

#include "NN-CLI_DataLoader.hpp"
#include "NN-CLI_TrainingProfiler.hpp"

#include <CNN_Core.hpp>
#include "Common/Common_TrainingMonitor.hpp"

#include <vector>

//===================================================================================================================//

namespace NN_CLI
{

  class CNNRunner : public Runner<CNN::Core<float>, CNN::CoreConfig<float>>
  {
    public:
      //-- Constructors --//
      CNNRunner(const QCommandLineParser& parser, LogLevel logLevel, IOConfig& ioConfig, AugmentationConfig& augConfig,
                std::unique_ptr<CNN::Core<float>>& core, CNN::CoreConfig<float>& coreConfig);

      //-- Mode methods --//
      int train();
      int test();
      int predict();

      //-- Accessors --//
      std::vector<std::string> getTimingLines(int maxWidth = 0) const override;

      ulong getNumOutputClasses() const override;

      //-- Model info overrides --//
      ulong getTotalParameters() const override;
      std::string getNetworkType() const override;
      std::string getInputShapeString() const override;
      ulong getNumConvLayers() const override;
      ulong getNumDenseLayers() const override;
      ulong getNumResidualBlocks() const override;

    protected:
      //-- Overrides --//
      void doSaveModel(const std::string& outputPath) override;

    private:
      //-- Sample loading --//
      std::pair<CNN::Samples<float>, bool> loadSamplesFromOptions(const std::string& modeName, QString& inputFilePath);

      //-- Training helpers --//
      void setupTrainingCallback(const QString& inputFilePath,
                                 std::shared_ptr<CNN::Core<float>> validationCore = nullptr,
                                 std::shared_ptr<Common::TrainingMonitor<float>> trainingMonitor = nullptr,
                                 const DataLoader<CNN::Sample<float>>* validationDataLoader = nullptr,
                                 const std::vector<ulong>* validationIndices = nullptr);

      //-- Per-phase timing profiler (fed by CNN's timing callback) --//
      TrainingProfiler profiler;
  };

} // namespace NN_CLI

#endif // NN_CLI_CNNRUNNER_HPP
