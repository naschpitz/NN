#include "test_helpers.hpp"

int testsPassed = 0;
int testsFailed = 0;

void runActvFuncTests();
void runUtilsTests();
void runCoreTests();
void runSerializationTests();
void runGPUTests();

int main() {
  std::cout << "========================================" << std::endl;
  std::cout << "       ANN Unit Tests" << std::endl;
  std::cout << "========================================" << std::endl;

  std::cout << "\n=== Activation Function Tests ===" << std::endl;
  runActvFuncTests();

  std::cout << "\n=== Utils / Device / Mode Tests ===" << std::endl;
  runUtilsTests();

  std::cout << "\n=== Core Tests ===" << std::endl;
  runCoreTests();

  std::cout << "\n=== Serialization Tests ===" << std::endl;
  runSerializationTests();

  std::cout << "\n=== GPU Tests ===" << std::endl;
  runGPUTests();

  std::cout << "\n========================================" << std::endl;
  std::cout << "Results: " << testsPassed << " passed, " << testsFailed << " failed" << std::endl;
  std::cout << "========================================" << std::endl;

  return testsFailed > 0 ? 1 : 0;
}

