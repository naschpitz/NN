#include "CNN_Core.hpp"
#include "CNN_CoreCPU.hpp"
#include "CNN_CoreGPU.hpp"

#include "Common/Common_Utils.hpp"

#include <stdexcept>

using namespace CNN;
using namespace Common;

//===================================================================================================================//

template <typename T>
std::unique_ptr<Core<T>> Core<T>::makeCore(const CoreConfig<T>& config)
{
  switch (config.deviceType) {
  case Common::DeviceType::CPU:
    return std::make_unique<CoreCPU<T>>(config);
  case Common::DeviceType::GPU:
    return std::make_unique<CoreGPU<T>>(config);
  default:
    throw std::runtime_error("Unknown device type");
  }
}

//===================================================================================================================//

template <typename T>
Core<T>::Core(const CoreConfig<T>& coreConfig) : coreConfig(coreConfig)
{
  this->sanityCheck(coreConfig);

  this->deviceType = coreConfig.deviceType;
  this->modeType = coreConfig.modeType;
  this->numThreads = coreConfig.numThreads;
  this->numGPUs = coreConfig.numGPUs;
  this->inputShape = coreConfig.inputShape;
  this->layersConfig = coreConfig.layersConfig;
  this->trainConfig = coreConfig.trainConfig;
  this->testConfig = coreConfig.testConfig;
  this->parameters = coreConfig.parameters;
  this->progressReports = coreConfig.progressReports;
  this->logLevel = coreConfig.logLevel;
}

//===================================================================================================================//

template <typename T>
void Core<T>::sanityCheck(const CoreConfig<T>& coreConfig)
{
  // Validate input shape
  if (coreConfig.inputShape.c == 0 || coreConfig.inputShape.h == 0 || coreConfig.inputShape.w == 0) {
    throw std::runtime_error("Invalid input shape: all dimensions must be > 0");
  }

  // Validate CNN layers shape compatibility
  coreConfig.layersConfig.validateShapes(coreConfig.inputShape);

  // Validate dense layers exist
  if (coreConfig.layersConfig.denseLayers.empty()) {
    throw std::runtime_error("At least one dense layer is required");
  }

  // Validate training config if in training mode
  if (coreConfig.modeType == Common::ModeType::TRAIN) {
    if (coreConfig.trainConfig.numEpochs == 0) {
      throw std::runtime_error("Number of epochs must be > 0 for training mode");
    }

    if (coreConfig.trainConfig.learningRate <= 0) {
      throw std::runtime_error("Learning rate must be > 0 for training mode");
    }
  }
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
  std::chrono::duration<double> elapsed = endTime - this->trainingStartTime;

  this->trainMetadata.endTime = Common::Utils::formatISO8601();
  this->trainMetadata.durationSeconds = elapsed.count();
  this->trainMetadata.durationFormatted = Common::Utils::formatDuration(elapsed.count());

  return this->trainMetadata;
}

//===================================================================================================================//

template <typename T>
PredictResults<T> Core<T>::predict(const Inputs<T>& inputs)
{
  // Wrap an in-memory Inputs<T> as a one-shot InputProvider and delegate
  // to the streaming overload. Per-input tensors are not deep-copied — the
  // provider just returns slices of the original vector.
  return predict(inputs.size(), [&inputs](ulong batchSize, ulong batchIndex) {
    ulong start = batchIndex * batchSize;
    ulong end = std::min(start + batchSize, static_cast<ulong>(inputs.size()));

    if (start >= end)
      return Inputs<T>{};

    return Inputs<T>(inputs.begin() + start, inputs.begin() + end);
  });
}

//===================================================================================================================//

template <typename T>
PredictResult<T> Core<T>::predict(const Input<T>& input)
{
  return predict(Inputs<T>{input})[0];
}

//===================================================================================================================//

template <typename T>
void Core<T>::predictStart()
{
  this->predictStartTime = std::chrono::system_clock::now();
}

//===================================================================================================================//

template <typename T>
PredictMetadata<T> Core<T>::predictEnd()
{
  auto endTime = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = endTime - this->predictStartTime;

  this->predictMetadata.startTime = Common::Utils::formatISO8601();
  this->predictMetadata.endTime = Common::Utils::formatISO8601();
  this->predictMetadata.durationSeconds = elapsed.count();
  this->predictMetadata.durationFormatted = Common::Utils::formatDuration(elapsed.count());

  return this->predictMetadata;
}

//===================================================================================================================//

// Explicit template instantiations
template class CNN::Core<int>;
template class CNN::Core<double>;
template class CNN::Core<float>;
