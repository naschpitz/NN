#include "test_helpers.hpp"

#include <QCoreApplication>

#include <cstring>

int testsPassed = 0;
int testsFailed = 0;
bool runFullTests = false;

void runANNTests();
void runCNNTests();
void runErrorTests();
void runDataLoaderTests();

int main(int argc, char* argv[]) {
  // Parse --full flag before QCoreApplication consumes argv
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--full") == 0) {
      runFullTests = true;
      break;
    }
  }

  QCoreApplication app(argc, argv);

  if (runFullTests) {
    std::cout << "Running ALL tests including full MNIST train+test (this may take a few minutes)." << std::endl;
  } else {
    std::cout << "Running quick tests only. Use --full to include MNIST train+test (may take a few minutes)." << std::endl;
  }
  std::cout << std::endl;

  std::cout << "=== ANN Tests ===" << std::endl;
  runANNTests();

  std::cout << std::endl;
  std::cout << "=== CNN Tests ===" << std::endl;
  runCNNTests();

  std::cout << std::endl;
  std::cout << "=== Error Handling Tests ===" << std::endl;
  runErrorTests();

  std::cout << std::endl;
  std::cout << "=== DataLoader Tests ===" << std::endl;
  runDataLoaderTests();

  // Cleanup temp files
  cleanupTemp();

  std::cout << std::endl;
  std::cout << "=== Results: " << testsPassed << " passed, " << testsFailed << " failed ===" << std::endl;
  return (testsFailed > 0) ? 1 : 0;
}

