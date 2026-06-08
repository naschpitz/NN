#pragma once

#include <sys/types.h>

namespace NN_Server
{

  struct InputConfig {
      bool isImage = false;
      ulong c = 0, h = 0, w = 0;

      bool hasShape() const
      {
        return c > 0 && h > 0 && w > 0;
      }
  };

  struct OutputConfig {
      bool isImage = false;
      ulong c = 0, h = 0, w = 0;

      bool hasShape() const
      {
        return c > 0 && h > 0 && w > 0;
      }
  };

} // namespace NN_Server
