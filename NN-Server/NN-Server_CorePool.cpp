#include "NN-Server_CorePool.hpp"
#include "NN-Server_Loader.hpp"

#include <iostream>
#include <stdexcept>

namespace NN_Server
{

  //===================================================================================================================//

  CorePool::CorePool(const std::string& configFilePath, int poolSize)
  {
    this->netType = Loader::detectNetworkType(configFilePath);
    this->outConfig = Loader::loadOutputConfig(configFilePath);
    std::string networkTypeStr = (this->netType == NetworkType::CNN) ? "CNN" : "ANN";

    std::cout << "Detected network type: " << networkTypeStr << "\n";
    std::cout << "Creating core pool with " << poolSize << " instance(s)...\n";

    this->entries.resize(static_cast<size_t>(poolSize));

    for (int i = 0; i < poolSize; ++i) {
      std::cout << "  Loading core " << (i + 1) << "/" << poolSize << "..." << std::flush;

      if (this->netType == NetworkType::ANN) {
        ANN::CoreConfig<float> config = Loader::loadANNConfig(configFilePath);
        this->entries[static_cast<size_t>(i)].annCore = ANN::Core<float>::makeCore(config);
      } else {
        CNN::CoreConfig<float> config = Loader::loadCNNConfig(configFilePath);

        // Store input shape from the first core (all cores share the same config)
        if (i == 0) {
          this->inC = config.inputShape.c;
          this->inH = config.inputShape.h;
          this->inW = config.inputShape.w;
        }

        this->entries[static_cast<size_t>(i)].cnnCore = CNN::Core<float>::makeCore(config);
      }

      this->entries[static_cast<size_t>(i)].available = true;
      std::cout << " done." << std::endl;
    }

    std::cout << "Core pool ready." << std::endl;
  }

  //===================================================================================================================//

  CoreHandle CorePool::acquire()
  {
    QMutexLocker locker(&this->mutex);

    // Wait until a core becomes available
    while (true) {
      for (size_t i = 0; i < this->entries.size(); ++i) {
        if (this->entries[i].available) {
          this->entries[i].available = false;

          CoreHandle handle;
          handle.index = static_cast<int>(i);
          handle.annCore = this->entries[i].annCore.get();
          handle.cnnCore = this->entries[i].cnnCore.get();

          return handle;
        }
      }

      this->condition.wait(&this->mutex);
    }
  }

  //===================================================================================================================//

  void CorePool::release(const CoreHandle& handle)
  {
    QMutexLocker locker(&this->mutex);

    this->entries[static_cast<size_t>(handle.index)].available = true;
    this->condition.wakeOne();
  }

  //===================================================================================================================//

} // namespace NN_Server

