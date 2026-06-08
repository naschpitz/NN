#pragma once

#include "_Core.hpp"
#include "_CoreConfig.hpp"
#include "_ActvFunc.hpp"
#include "Common/Common_Device.hpp"
#include "Common/Common_Mode.hpp"
#include "_LayersConfig.hpp"
#include "_Types.hpp"
#include "_Parameters.hpp"
#include "_Sample.hpp"
#include "Common/Common_TrainingConfig.hpp"
#include "Common/Common_TrainingProgress.hpp"
#include "Common/Common_TrainingMetadata.hpp"
#include "Common/Common_PredictMetadata.hpp"
#include "Common/Common_TestResult.hpp"
#include "_Utils.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include <stdexcept>
#include <functional>

extern int testsPassed;
extern int testsFailed;

// clang-format off
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
// clang-format on

inline ANN::LayersConfig makeLayersConfig(std::initializer_list<ANN::Layer> layers)
{
  ANN::LayersConfig config;

  for (const auto& layer : layers) {
    config.push_back(layer);
  }

  return config;
}
