#include "test_helpers.hpp"

int testsPassed = 0;
int testsFailed = 0;

void runActvFuncTests();
void runUtilsTests();
void runCPUBasicTests();
void runCPUBasicTests2();
void runCPUFeaturesTests();
void runCPUFeaturesTests2();
void runCPUExactTests();
void runCPUExactTests2();
void runSerializationTests();
void runGPUBasicTests();
void runGPUBasicTests2();
void runGPUFeaturesTests();
void runGPUMultiGPUTests();
void runGPUExactTests();

int main()
{
  std::cout << "========================================" << std::endl;
  std::cout << "        Unit Tests" << std::endl;
  std::cout << "========================================" << std::endl;

  std::cout << "\n=== Activation Function Tests ===" << std::endl;
  runActvFuncTests();

  std::cout << "\n=== Utils / Device / Mode Tests ===" << std::endl;
  runUtilsTests();

  std::cout << "\n=== Core CPU Tests ===" << std::endl;
  runCPUBasicTests();
  runCPUBasicTests2();
  runCPUFeaturesTests();
  runCPUFeaturesTests2();
  runCPUExactTests();
  runCPUExactTests2();

  std::cout << "\n=== Core GPU Tests ===" << std::endl;
  runGPUBasicTests();
  runGPUBasicTests2();
  runGPUFeaturesTests();
  runGPUMultiGPUTests();
  runGPUExactTests();

  std::cout << "\n=== Serialization Tests ===" << std::endl;
  runSerializationTests();

  std::cout << "\n========================================" << std::endl;
  std::cout << "=== Results: " << testsPassed << " passed, " << testsFailed << " failed ===" << std::endl;
  std::cout << "========================================" << std::endl;

  return testsFailed > 0 ? 1 : 0;
}
