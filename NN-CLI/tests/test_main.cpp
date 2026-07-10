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
void runGPUValidationTests();

// CNN tests
void runCNNCPUBasicTests();
void runCNNCPUFeatureTests();
void runCNNCPUSaveLoadTests();
void runCNNDenseRoundTripTests();
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
void runGPUAugmentTests();
void runTerminalUITests();
void runCalibrateControllerTests();
void runLearningRateSchedulerTests();

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

  std::cout << "=== ANN CPU Tests ===" << std::endl;
  runCPUBasicTests();
  runCPUMNISTTests();
  runCPUFeatureTests();
  runANNCPUSaveLoadTests();

  std::cout << std::endl;
  std::cout << "=== ANN GPU Tests ===" << std::endl;
  runGPUMNISTTests();
  runGPUValidationTests();

  std::cout << std::endl;
  std::cout << "=== CNN CPU Tests ===" << std::endl;
  runCNNCPUBasicTests();
  runCNNCPUFeatureTests();
  runCNNCPUSaveLoadTests();
  runCNNDenseRoundTripTests();
  runCNNCPUPredictTests();

  std::cout << std::endl;
  std::cout << "=== CNN GPU Tests ===" << std::endl;
  runCNNGPULayerTests();
  runCNNGPUDiagnosticTests();
  runCNNGPUISICTests();
  runCNNGPUMNISTTests();
  runCNNGPUSaveLoadTests();

  std::cout << std::endl;
  std::cout << "=== Augmentation Tests ===" << std::endl;
  runGPUAugmentTests();

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

  std::cout << std::endl;
  std::cout << "=== Calibrate Controller Tests ===" << std::endl;
  runCalibrateControllerTests();

  std::cout << std::endl;
  std::cout << "=== LR Scheduler Tests ===" << std::endl;
  runLearningRateSchedulerTests();

  // Cleanup temp files
  cleanupTemp();

  std::cout << std::endl;
  std::cout << "=== Results: " << testsPassed << " passed, " << testsFailed << " failed ===" << std::endl;
  return (testsFailed > 0) ? 1 : 0;
}
