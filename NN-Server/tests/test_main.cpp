#include "test_helpers.hpp"

#include <QCoreApplication>

int testsPassed = 0;
int testsFailed = 0;

void runEndpointTests();

int main(int argc, char* argv[])
{
  QCoreApplication app(argc, argv);

  std::cout << "=== NN-Server Endpoint Integration Tests ===" << std::endl;
  std::cout << std::endl;

  runEndpointTests();

  std::cout << std::endl;
  std::cout << "=== Results: " << testsPassed << " passed, " << testsFailed << " failed ===" << std::endl;
  return (testsFailed > 0) ? 1 : 0;
}

