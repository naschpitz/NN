#ifndef NN_CLI_ANNRUNNER_HPP
#define NN_CLI_ANNRUNNER_HPP

#include "NN-CLI_Runner.hpp"

#include "NN-CLI_DataLoader.hpp"

#include <ANN_Core.hpp>
#include "Common/Common_TrainingMonitor.hpp"

#include <vector>

//===================================================================================================================//

namespace NN_CLI
{

  class ANNRunner : public Runner<ANN::Core<float>, ANN::CoreConfig<float>>
  {
    public:
      //-- Constructors --//
      ANNRunner(const QCommandLineParser& parser, LogLevel logLevel, IOConfig& ioConfig, AugmentationConfig& augConfig,
                std::unique_ptr<ANN::Core<float>>& core, ANN::CoreConfig<float>& coreConfig);

      //-- Mode methods --//
      int train();
      int test();
      int predict();

      //-- Accessors --//
      std::vector<std::string> getTimingLines(int maxWidth = 0) const override;

    protected:
      //-- Overrides --//
      void doSaveModel(const std::string& outputPath) override;

    private:
      //-- Sample loading --//
      std::pair<ANN::Samples<float>, bool> loadSamplesFromOptions(const std::string& modeName, QString& inputFilePath);

      //-- Training helpers --//
      void setupTrainingCallback(const QString& inputFilePath,
                                 std::shared_ptr<ANN::Core<float>> validationCore = nullptr,
                                 std::shared_ptr<Common::TrainingMonitor<float>> trainingMonitor = nullptr,
                                 const DataLoader<ANN::Sample<float>>* validationDataLoader = nullptr,
                                 const std::vector<ulong>* validationIndices = nullptr);
  };

} // namespace NN_CLI

#endif // NN_CLI_ANNRUNNER_HPP
