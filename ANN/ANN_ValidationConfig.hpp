#ifndef ANN_VALIDATIONCONFIG_HPP
#define ANN_VALIDATIONCONFIG_HPP

#include <sys/types.h>

//===================================================================================================================//

namespace ANN
{
  struct ValidationConfig {
    bool enabled = true;
    bool autoSize = true;
    float size = 0.15f;
    ulong checkInterval = 1;
  };
}

//===================================================================================================================//

#endif // ANN_VALIDATIONCONFIG_HPP
