#include "NN-CLI_TuiState.hpp"

#include <atomic>

namespace NN_CLI
{
  namespace
  {
    std::atomic<bool> g_tuiActive{false};
  }

  bool isTuiActive() noexcept
  {
    return g_tuiActive.load(std::memory_order_acquire);
  }

  void setTuiActive(bool active) noexcept
  {
    g_tuiActive.store(active, std::memory_order_release);
  }
}
