#ifndef NN_SERVER_COREPOOL_HPP
#define NN_SERVER_COREPOOL_HPP

#include "NN-Server_CoreHandle.hpp"
#include "NN-Server_IOConfig.hpp"
#include "NN-Server_Loader.hpp"
#include "NN-Server_NetworkType.hpp"

#include <ANN_Core.hpp>
#include <CNN_Core.hpp>

#include <QMutex>
#include <QWaitCondition>

#include <memory>
#include <vector>

namespace NN_Server
{

  /**
   * Thread-safe pool of pre-loaded Core objects.
   * Each Core is completely independent (owns its own weights and worker buffers),
   * so concurrent predictions on different Cores are fully isolated.
   */
  class CorePool
  {
    public:
      CorePool(const std::string& configFilePath, int poolSize);

      // Acquire an available Core. Blocks until one is free.
      CoreHandle acquire();

      // Release a Core back to the pool.
      void release(const CoreHandle& handle);

      NetworkType networkType() const
      {
        return this->netType;
      }

      int size() const
      {
        return static_cast<int>(this->entries.size());
      }

      const InputConfig& inputConfig() const
      {
        return this->inConfig;
      }

      const OutputConfig& outputConfig() const
      {
        return this->outConfig;
      }

    private:
      struct CoreEntry {
          std::unique_ptr<ANN::Core<float>> annCore;
          std::unique_ptr<CNN::Core<float>> cnnCore;
          bool available = true;
      };

      std::vector<CoreEntry> entries;
      NetworkType netType;
      InputConfig inConfig;
      OutputConfig outConfig;

      QMutex mutex;
      QWaitCondition condition;
  };

} // namespace NN_Server

#endif // NN_SERVER_COREPOOL_HPP
