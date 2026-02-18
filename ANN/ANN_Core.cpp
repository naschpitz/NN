#include "ANN_Core.hpp"
#include "ANN_CoreCPU.hpp"
#include "ANN_CoreGPU.hpp"

#include <OCLW_Core.hpp>
#include <QFile>

using namespace ANN;

//===================================================================================================================//

template <typename T>
Core<T>::Core(const CoreConfig<T>& coreConfig) {
  this->coreTypeType = coreConfig.coreTypeType;
  this->coreModeType = coreConfig.coreModeType;

  this->layersConfig = coreConfig.layersConfig;
  this->trainingConfig = coreConfig.trainingConfig;
  this->parameters = coreConfig.parameters;
}

//===================================================================================================================//

template <typename T>
std::unique_ptr<Core<T>> Core<T>::makeCore(const CoreConfig<T>& coreConfig) {
  switch (coreConfig.coreTypeType) {
    case CoreTypeType::CPU:
      return std::make_unique<CoreCPU<T>>(coreConfig);
    case CoreTypeType::GPU:
    default:
      return std::make_unique<CoreGPU<T>>(coreConfig);
  }
}

//===================================================================================================================//

template <typename T>
void Core<T>::sanityCheck(const CoreConfig<T>& coreConfig) {
  if (coreConfig.coreTypeType == CoreTypeType::UNKNOWN) {
    throw std::runtime_error("Unkown coreTypeType");
  }

  if (coreConfig.coreModeType == CoreModeType::UNKNOWN) {
    throw std::runtime_error("Unkown coreModeType");
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

// (Optional) Explicit template instantiations.
template class ANN::Core<int>;
template class ANN::Core<double>;
template class ANN::Core<float>;
