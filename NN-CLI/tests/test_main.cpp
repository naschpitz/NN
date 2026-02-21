#include "test_helpers.hpp"

#include <QCoreApplication>

int testsPassed = 0;
int testsFailed = 0;

void runANNTests();
void runCNNTests();
void runErrorTests();

int main(int argc, char* argv[]) {
  QCoreApplication app(argc, argv);

  std::cout << "=== ANN Tests ===" << std::endl;
  runANNTests();

  std::cout << std::endl;
  std::cout << "=== CNN Tests ===" << std::endl;
  runCNNTests();

  std::cout << std::endl;
  std::cout << "=== Error Handling Tests ===" << std::endl;
  runErrorTests();

  // Cleanup temp files
  cleanupTemp();

  std::cout << std::endl;
  std::cout << "=== Results: " << testsPassed << " passed, " << testsFailed << " failed ===" << std::endl;
  return (testsFailed > 0) ? 1 : 0;
}

