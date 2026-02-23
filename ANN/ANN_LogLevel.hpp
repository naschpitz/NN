#ifndef ANN_LOGLEVEL_HPP
#define ANN_LOGLEVEL_HPP

//===================================================================================================================//

namespace ANN {
  enum class LogLevel : int {
    QUIET   = 0,
    ERROR   = 1,
    WARNING = 2,
    INFO    = 3,
    DEBUG   = 4
  };
}

//===================================================================================================================//

#endif // ANN_LOGLEVEL_HPP

