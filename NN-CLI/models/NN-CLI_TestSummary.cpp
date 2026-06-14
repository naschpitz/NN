#include "NN-CLI_TestSummary.hpp"
#include "NN-CLI_SummaryTable.hpp"
#include "NN-CLI_TrainingSummary.hpp"

#include <string>
#include <variant>
#include <vector>

namespace NN_CLI
{

  //===================================================================================================================//

  void TestSummary::printCNN(const CNN::CoreConfig<float>& cnnConfig, ulong testSamples)
  {
    const auto& inputShape = cnnConfig.inputShape;
    const auto& layers = cnnConfig.layersConfig;
    const auto& costConfig = cnnConfig.costFunctionConfig;

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

    std::string deviceStr = SummaryTable::deviceString(cnnConfig.deviceType, cnnConfig.numGPUs, cnnConfig.numThreads);

    std::string costStr;

    switch (costConfig.type) {
    case Common::CostFunctionType::CROSS_ENTROPY:
      costStr = "Cross-entropy";
      break;
    case Common::CostFunctionType::SQUARED_DIFFERENCE:
      costStr = "Squared difference";
      break;
    case Common::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE:
      costStr = "Weighted squared difference";
      break;
    }

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
    rows.push_back({"", ""});
    rows.push_back({"Test samples", SummaryTable::formatWithCommas(testSamples)});
    rows.push_back({"Batch size", std::to_string(cnnConfig.testConfig.batchSize)});
    rows.push_back({"Cost function", costStr});

    SummaryTable::print("Test Configuration", rows);
  }

  //===================================================================================================================//

  void TestSummary::print(const ANN::CoreConfig<float>& annConfig, ulong testSamples)
  {
    ulong denseCount = annConfig.layersConfig.size();

    std::string deviceStr = SummaryTable::deviceString(annConfig.deviceType, annConfig.numGPUs, annConfig.numThreads);

    std::string costStr;

    switch (annConfig.costFunctionConfig.type) {
    case Common::CostFunctionType::CROSS_ENTROPY:
      costStr = "Cross-entropy";
      break;
    case Common::CostFunctionType::SQUARED_DIFFERENCE:
      costStr = "Squared difference";
      break;
    case Common::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE:
      costStr = "Weighted squared difference";
      break;
    }

    std::vector<SummaryRow> rows;
    rows.push_back({"Device", deviceStr});
    rows.push_back({"Network type", ""});
    rows.push_back({"", ""});
    rows.push_back({"Dense layers", std::to_string(denseCount)});
    rows.push_back({"", ""});
    rows.push_back({"Test samples", SummaryTable::formatWithCommas(testSamples)});
    rows.push_back({"Batch size", std::to_string(annConfig.testConfig.batchSize)});
    rows.push_back({"Cost function", costStr});

    SummaryTable::print("Test Configuration", rows);
  }

} // namespace NN_CLI
