#include "test_helpers.hpp"

#include <QCoreApplication>

#include <cstring>

int testsPassed = 0;
int testsFailed = 0;
bool runFullTests = false;

//  tests
void runCPUBasicTests();
void runCPUMNISTTests();
void runCPUFeatureTests();
void runANNCPUSaveLoadTests();
void runGPUMNISTTests();

// CNN tests
void runCNNCPUBasicTests();
void runCNNCPUFeatureTests();
void runCNNCPUSaveLoadTests();
void runCNNCPUPredictTests();
void runCNNGPULayerTests();
void runCNNGPUDiagnosticTests();
void runCNNGPUISICTests();
void runCNNGPUMNISTTests();
void runCNNGPUSaveLoadTests();

// Other tests
void runErrorTests();
void runDataLoaderTests();
void runValidationTests();
void runMonitoringTests();
void runGpuAugmentTests();
void runTerminalUITests();

int main(int argc, char* argv[])
{
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
    std::cout << "Running quick tests only. Use --full to include MNIST train+test (may take a few minutes)."
              << std::endl;
  }

  std::cout << std::endl;

  std::cout << "===  CPU Tests ===" << std::endl;
  runCPUBasicTests();
  runCPUMNISTTests();
  runCPUFeatureTests();
  runANNCPUSaveLoadTests();

  std::cout << std::endl;
  std::cout << "===  GPU Tests ===" << std::endl;
  runGPUMNISTTests();

  std::cout << std::endl;
  std::cout << "=== CNN CPU Tests ===" << std::endl;
  runCNNCPUBasicTests();
  runCNNCPUFeatureTests();
  runCNNCPUSaveLoadTests();
  runCNNCPUPredictTests();

  std::cout << std::endl;
  std::cout << "=== CNN GPU Tests ===" << std::endl;
  runCNNGPULayerTests();
  runCNNGPUDiagnosticTests();
  runCNNGPUISICTests();
  runCNNGPUMNISTTests();
  runCNNGPUSaveLoadTests();

  std::cout << std::endl;
  std::cout << "=== GPU Augmentation Tests ===" << std::endl;
  runGpuAugmentTests();

  std::cout << std::endl;
  std::cout << "=== Error Handling Tests ===" << std::endl;
  runErrorTests();

  std::cout << std::endl;
  std::cout << "=== DataLoader Tests ===" << std::endl;
  runDataLoaderTests();

  std::cout << std::endl;
  std::cout << "=== Validation Split Tests ===" << std::endl;
  runValidationTests();

  std::cout << std::endl;
  std::cout << "=== Monitoring Tests ===" << std::endl;
  runMonitoringTests();

  std::cout << std::endl;
  std::cout << "=== Terminal UI Tests ===" << std::endl;
  runTerminalUITests();

  // Cleanup temp files
  cleanupTemp();

  std::cout << std::endl;
  std::cout << "=== Results: " << testsPassed << " passed, " << testsFailed << " failed ===" << std::endl;
  return (testsFailed > 0) ? 1 : 0;
}
