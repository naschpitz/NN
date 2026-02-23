#include "ANN_Core.hpp"
#include "ANN_CoreCPU.hpp"
#include "ANN_CoreGPU.hpp"
#include "ANN_Utils.hpp"

#include <OCLW_Core.hpp>
#include <QFile>

using namespace ANN;

//===================================================================================================================//

template <typename T>
Core<T>::Core(const CoreConfig<T>& coreConfig) {
  this->deviceType = coreConfig.deviceType;
  this->modeType = coreConfig.modeType;

  this->layersConfig = coreConfig.layersConfig;
  this->trainingConfig = coreConfig.trainingConfig;
  this->parameters = coreConfig.parameters;
  this->progressReports = coreConfig.progressReports;
  this->logLevel = coreConfig.logLevel;
}

//===================================================================================================================//

template <typename T>
std::unique_ptr<Core<T>> Core<T>::makeCore(const CoreConfig<T>& coreConfig) {
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
void Core<T>::sanityCheck(const CoreConfig<T>& coreConfig) {
  if (coreConfig.deviceType == DeviceType::UNKNOWN) {
    throw std::runtime_error("Unknown deviceType");
  }

  if (coreConfig.modeType == ModeType::UNKNOWN) {
    throw std::runtime_error("Unknown modeType");
  }
}

//===================================================================================================================//

template <typename T>
T Core<T>::calculateLoss(const Output<T>& expected) {
  ulong numLayers = this->layersConfig.size();
  const Output<T>& actual = this->actvs[numLayers - 1];

  T loss = 0;
  
  for (ulong i = 0; i < expected.size(); i++) {
    T diff = actual[i] - expected[i];
    loss += diff * diff;
  }

  return loss / static_cast<T>(expected.size());
}

//===================================================================================================================//

template <typename T>
void Core<T>::trainingStart(ulong numSamples) {
  this->trainingStartTime = std::chrono::system_clock::now();
  this->trainingMetadata.startTime = Utils<T>::formatISO8601();
  this->trainingMetadata.numSamples = numSamples;
}

//===================================================================================================================//

template <typename T>
TrainingMetadata<T> Core<T>::trainingEnd() {
  auto endTime = std::chrono::system_clock::now();
  this->trainingMetadata.endTime = Utils<T>::formatISO8601();

  std::chrono::duration<double> duration = endTime - this->trainingStartTime;
  this->trainingMetadata.durationSeconds = duration.count();
  this->trainingMetadata.durationFormatted = Utils<T>::formatDuration(this->trainingMetadata.durationSeconds);

  return this->trainingMetadata;
}

//===================================================================================================================//

template <typename T>
void Core<T>::predictStart() {
  this->predictStartTime = std::chrono::system_clock::now();
  this->predictMetadata.startTime = Utils<T>::formatISO8601();
}

//===================================================================================================================//

template <typename T>
PredictMetadata<T> Core<T>::predictEnd() {
  auto endTime = std::chrono::system_clock::now();
  this->predictMetadata.endTime = Utils<T>::formatISO8601();

  std::chrono::duration<double> duration = endTime - this->predictStartTime;
  this->predictMetadata.durationSeconds = duration.count();
  this->predictMetadata.durationFormatted = Utils<T>::formatDuration(this->predictMetadata.durationSeconds);

  return this->predictMetadata;
}

//===================================================================================================================//

// (Optional) Explicit template instantiations.
template class ANN::Core<int>;
template class ANN::Core<double>;
template class ANN::Core<float>;
