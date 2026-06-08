#ifndef NN_CLI_VALIDATIONDATASETCONFIG_HPP
#define NN_CLI_VALIDATIONDATASETCONFIG_HPP

#include "NN-CLI_Types.hpp"

namespace NN_CLI
{

  struct ValidationDatasetConfig {
      bool enabled = true; // Enable validation split
      bool autoSize = true; // Auto-select split ratio based on dataset size
      float size = 0.15f; // Fixed validation fraction (used when autoSize is false)
      ulong checkInterval = 1; // Run validation every N epochs (1 = every epoch)
  };

} // namespace NN_CLI

#endif // NN_CLI_VALIDATIONDATASETCONFIG_HPP
