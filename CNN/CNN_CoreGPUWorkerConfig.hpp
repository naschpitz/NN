#ifndef CNN_COREGPUWORKERCONFIG_HPP
#define CNN_COREGPUWORKERCONFIG_HPP

#include "CNN_CoreConfig.hpp"
#include "Common/Common_LogLevel.hpp"
#include "CNN_LayersConfig.hpp"
#include "Common/Common_CostFunctionConfig.hpp"
#include "Common/Common_TrainingConfig.hpp"
#include "CNN_Parameters.hpp"
#include "CNN_Types.hpp"

//===================================================================================================================//

namespace CNN
{
  using namespace Common;
  // Configuration for a single GPU worker. Extracts only the fields needed
  // from CoreConfig and adds the per-GPU batch size.
  template <typename T>
  class CoreGPUWorkerConfig
  {
    public:
      //-- Constructor --//
      explicit CoreGPUWorkerConfig(const CoreConfig<T>& config);

      //-- CNN topology --//
      Shape3D inputShape;
      LayersConfig layersConfig;

      //-- Training --//
      Common::TrainingConfig<T> trainingConfig;
      Common::CostFunctionConfig<T> costFunctionConfig;

      //-- Parameters (initial weights) --//
      Parameters<T> parameters;

      //-- Per-GPU batch size --//
      ulong batchSize = 1;

      //-- Logging / progress --//
      ulong progressReports = 1000;
      Common::LogLevel logLevel = Common::LogLevel::ERROR;
  };
}

//===================================================================================================================//

#endif // CNN_COREGPUWORKERCONFIG_HPP
