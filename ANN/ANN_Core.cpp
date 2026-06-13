#include "ANN_Core.hpp"
#include "ANN_CoreCPU.hpp"
#include "ANN_CoreGPU.hpp"
#include "Common/Common_Utils.hpp"

using namespace ANN;
using namespace Common;

//===================================================================================================================//

template <typename T>
Core<T>::Core(const CoreConfig<T>& coreConfig)
{
  this->deviceType = coreConfig.deviceType;
  this->modeType = coreConfig.modeType;
  this->numThreads = coreConfig.numThreads;
  this->numGPUs = coreConfig.numGPUs;

  this->layersConfig = coreConfig.layersConfig;
  this->costFunctionConfig = coreConfig.costFunctionConfig;
  this->trainConfig = coreConfig.trainConfig;
  this->testConfig = coreConfig.testConfig;
  this->parameters = coreConfig.parameters;
  this->progressReports = coreConfig.progressReports;
  this->logLevel = coreConfig.logLevel;
}

//===================================================================================================================//

template <typename T>
std::unique_ptr<Core<T>> Core<T>::makeCore(const CoreConfig<T>& coreConfig)
{
  switch (coreConfig.deviceType) {
  case DeviceType::CPU:
    return std::make_unique<CoreCPU<T>>(coreConfig);
  case DeviceType::GPU:
  default:
    return std::make_unique<CoreGPU<T>>(coreConfig);
  }
}

//===================================================================================================================//

template <typename T>
void Core<T>::sanityCheck(const CoreConfig<T>& coreConfig)
{
  // deviceType and modeType are validated at parse time by nameToType (throws on invalid)
  // No additional checks needed here
}

//===================================================================================================================//

template <typename T>
void Core<T>::trainingStart(ulong numSamples)
{
  this->trainingStartTime = std::chrono::system_clock::now();
  this->trainMetadata.startTime = Common::Utils::formatISO8601();
  this->trainMetadata.numSamples = numSamples;
}

//===================================================================================================================//

template <typename T>
TrainMetadata<T> Core<T>::trainingEnd()
{
  auto endTime = std::chrono::system_clock::now();
  this->trainMetadata.endTime = Common::Utils::formatISO8601();

  std::chrono::duration<double> duration = endTime - this->trainingStartTime;
  this->trainMetadata.durationSeconds = duration.count();
  this->trainMetadata.durationFormatted = Common::Utils::formatDuration(this->trainMetadata.durationSeconds);

  return this->trainMetadata;
}

//===================================================================================================================//

template <typename T>
PredictResult<T> Core<T>::predict(const Input<T>& input)
{
  // One-shot provider for the single sample.
  PredictResults<T> results = predict(1, [&input](ulong, ulong batchIndex) {
    if (batchIndex > 0)
      return Inputs<T>{};

    return Inputs<T>{input};
  });

  return results[0];
}

//===================================================================================================================//

template <typename T>
void Core<T>::predictStart()
{
  this->predictStartTime = std::chrono::system_clock::now();
  this->predictMetadata.startTime = Common::Utils::formatISO8601();
}

//===================================================================================================================//

template <typename T>
PredictMetadata<T> Core<T>::predictEnd()
{
  auto endTime = std::chrono::system_clock::now();
  this->predictMetadata.endTime = Common::Utils::formatISO8601();

  std::chrono::duration<double> duration = endTime - this->predictStartTime;
  this->predictMetadata.durationSeconds = duration.count();
  this->predictMetadata.durationFormatted = Common::Utils::formatDuration(this->predictMetadata.durationSeconds);

  return this->predictMetadata;
}

//===================================================================================================================//

// (Optional) Explicit template instantiations.
template class ANN::Core<int>;
template class ANN::Core<double>;
template class ANN::Core<float>;
