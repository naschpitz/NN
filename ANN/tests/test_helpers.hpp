#pragma once

#include "ANN_Core.hpp"
#include "ANN_CoreConfig.hpp"
#include "ANN_ActvFunc.hpp"
#include "ANN_Device.hpp"
#include "ANN_Mode.hpp"
#include "ANN_LayersConfig.hpp"
#include "ANN_Types.hpp"
#include "ANN_Parameters.hpp"
#include "ANN_Sample.hpp"
#include "ANN_TrainingConfig.hpp"
#include "ANN_TrainingProgress.hpp"
#include "ANN_TrainingMetadata.hpp"
#include "ANN_PredictMetadata.hpp"
#include "ANN_TestResult.hpp"
#include "ANN_Utils.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include <stdexcept>
#include <functional>

extern int testsPassed;
extern int testsFailed;

#define CHECK(cond, msg) do { \
  if (!(cond)) { \
    std::cerr << "FAIL: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
    testsFailed++; \
  } else { \
    testsPassed++; \
  } \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) CHECK(std::fabs((a) - (b)) < (tol), msg)

#define CHECK_THROWS(expr, msg) do { \
  bool threw = false; \
  try { expr; } catch (...) { threw = true; } \
  CHECK(threw, msg); \
} while(0)

inline ANN::LayersConfig makeLayersConfig(std::initializer_list<ANN::Layer> layers) {
  ANN::LayersConfig config;
  for (const auto& layer : layers) {
    config.push_back(layer);
  }
  return config;
}

