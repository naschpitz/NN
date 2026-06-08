#ifndef COMMON_VALIDATIONCONFIG_HPP
#define COMMON_VALIDATIONCONFIG_HPP

#include <sys/types.h>

//===================================================================================================================//

namespace Common
{
  struct ValidationConfig {
      bool enabled = true;
      bool autoSize = true;
      float size = 0.15f;
      ulong checkInterval = 1;
  };
}

//===================================================================================================================//

#endif // COMMON_VALIDATIONCONFIG_HPP
