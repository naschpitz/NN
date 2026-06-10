#include "NN-CLI_PredictSummary.hpp"
#include "NN-CLI_SummaryTable.hpp"
#include "NN-CLI_TrainingSummary.hpp"

#include <string>
#include <variant>
#include <vector>

namespace NN_CLI
{

  //===================================================================================================================//

  void PredictSummary::printANN(const ANN::CoreConfig<float>& annConfig, ulong numInputs, const std::string& inputPath,
                                 const std::string& outputPath)
  {
    ulong denseCount = annConfig.layersConfig.size();
    ulong outputNeurons = annConfig.layersConfig.empty() ? 0 : annConfig.layersConfig.back().numNeurons;

    std::string deviceStr = SummaryTable::deviceString(annConfig.deviceType, annConfig.numGPUs, annConfig.numThreads);

    std::vector<SummaryRow> rows;
    rows.push_back({"Device", deviceStr});
    rows.push_back({"Network type", ""});
    rows.push_back({"", ""});
    rows.push_back({"Dense layers", std::to_string(denseCount)});
    rows.push_back({"Output neurons", std::to_string(outputNeurons)});
    rows.push_back({"", ""});
    rows.push_back({"Inputs", SummaryTable::formatWithCommas(numInputs)});
    rows.push_back({"Input file", inputPath});
    rows.push_back({"Output file", outputPath});

    SummaryTable::print("Predict Configuration", rows);
  }

  //===================================================================================================================//

  void PredictSummary::printCNN(const CNN::CoreConfig<float>& cnnConfig, ulong numInputs, const std::string& inputPath,
                                 const std::string& outputPath)
  {
    const auto& inputShape = cnnConfig.inputShape;
    const auto& layers = cnnConfig.layersConfig;

    ulong convCount = 0;
    ulong residualCount = 0;

    for (const auto& l : layers.cnnLayers) {
      if (l.type == CNN::LayerType::CONV)
        convCount++;
      else if (l.type == CNN::LayerType::RESIDUAL_START)
        residualCount++;
    }

    ulong denseCount = layers.denseLayers.size();
    ulong totalParams = TrainingSummary::countCNNParameters(cnnConfig);
    ulong outputNeurons = layers.denseLayers.empty() ? 0 : layers.denseLayers.back().numNeurons;

    std::string deviceStr = SummaryTable::deviceString(cnnConfig.deviceType, cnnConfig.numGPUs, cnnConfig.numThreads);

    std::string inputShapeStr =
      std::to_string(inputShape.c) + " x " + std::to_string(inputShape.h) + " x " + std::to_string(inputShape.w);

    std::vector<SummaryRow> rows;
    rows.push_back({"Device", deviceStr});
    rows.push_back({"Input shape", inputShapeStr});
    rows.push_back({"Network type", "CNN"});
    rows.push_back({"", ""});
    rows.push_back({"Conv layers", std::to_string(convCount)});
    rows.push_back({"Dense layers", std::to_string(denseCount)});
    rows.push_back({"Residual blocks", std::to_string(residualCount)});
    rows.push_back({"Total parameters", SummaryTable::formatWithCommas(totalParams)});
    rows.push_back({"Output neurons", std::to_string(outputNeurons)});
    rows.push_back({"", ""});
    rows.push_back({"Inputs", SummaryTable::formatWithCommas(numInputs)});
    rows.push_back({"Input file", inputPath});
    rows.push_back({"Output file", outputPath});

    SummaryTable::print("Predict Configuration", rows);
  }

} // namespace NN_CLI
