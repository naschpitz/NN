#include "test_helpers.hpp"

int testsPassed = 0;
int testsFailed = 0;

void runCPUConv2DTests();
void runCPULayerTests();
void runCPULayerTests2();
void runCPUInstanceNormTests();
void runBatchNormTests();
void runCPUGlobalAvgPoolTests();
void runCPUGlobalDualPoolTests();
void runCPUResidualTests();
void runCPUIntegrationBasicTests();
void runCPUIntegrationBasicTests2();
void runCPUIntegrationBasicTests3();
void runCPUIntegrationGlobalPoolTests();
void runCPUIntegrationResidualTests();
void runCPUIntegrationCostFuncTests();
void runCPUIntegrationExactTests();
void runCPUIntegrationBatchNormTests();
void runGPUBasicTests();
void runGPUBasicTests2();
void runGPUBasicTests3();
void runGPUBasicTests4();
void runGPUGlobalPoolTests();
void runGPUMultiGPUTests();
void runGPUMultiGPUTests2();
void runGPUExactTests();
void runGPUExactGlobalPoolTests();
void runGPUExactBNTests();
void runGPUBatchNormTests();
void runGPUBatchNormTests2();
void runCPUPredictStopTests();
void runCPUTrainStopTests();
void runMultiThreadTests();
void runGPUPredictStopTests();
void runGPUTrainStopTests();

int main()
{
  std::cout << "=== CNN CPU Unit Tests ===" << std::endl;
  runCPUConv2DTests();
  runCPULayerTests();
  runCPULayerTests2();
  runCPUInstanceNormTests();
  runBatchNormTests();
  runCPUGlobalAvgPoolTests();
  runCPUGlobalDualPoolTests();
  runCPUResidualTests();

  std::cout << std::endl;
  std::cout << "=== CNN CPU Integration Tests ===" << std::endl;
  runCPUIntegrationBasicTests();
  runCPUIntegrationBasicTests2();
  runCPUIntegrationBasicTests3();
  runCPUIntegrationGlobalPoolTests();
  runCPUIntegrationResidualTests();
  runCPUIntegrationCostFuncTests();
  runCPUIntegrationExactTests();
  runCPUIntegrationBatchNormTests();

  std::cout << std::endl;
  std::cout << "=== CNN GPU Tests ===" << std::endl;
  runGPUBasicTests();
  runGPUBasicTests2();
  runGPUBasicTests3();
  runGPUBasicTests4();
  runGPUGlobalPoolTests();
  runGPUMultiGPUTests();
  runGPUMultiGPUTests2();
  runGPUExactTests();
  runGPUExactGlobalPoolTests();
  runGPUExactBNTests();
  runGPUBatchNormTests();
  runGPUBatchNormTests2();

  std::cout << std::endl;
  std::cout << "=== CNN Predict Stop Tests ===" << std::endl;
  runCPUPredictStopTests();

  std::cout << std::endl;
  std::cout << "=== CNN GPU Predict Stop Tests ===" << std::endl;
  runGPUPredictStopTests();

  std::cout << std::endl;
  std::cout << "=== CNN Train Stop Tests ===" << std::endl;
  runCPUTrainStopTests();
  runMultiThreadTests();

  std::cout << std::endl;
  std::cout << "=== CNN GPU Train Stop Tests ===" << std::endl;
  runGPUTrainStopTests();

  std::cout << std::endl;
  std::cout << "=== Results: " << testsPassed << " passed, " << testsFailed << " failed ===" << std::endl;
  return (testsFailed > 0) ? 1 : 0;
}
