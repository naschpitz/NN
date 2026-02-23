#ifndef NN_CLI_LOGLEVEL_HPP
#define NN_CLI_LOGLEVEL_HPP

//===================================================================================================================//

namespace NN_CLI {
  enum class LogLevel : int {
    QUIET   = 0,
    ERROR   = 1,
    WARNING = 2,
    INFO    = 3,
    DEBUG   = 4
  };
}

//===================================================================================================================//

#endif // NN_CLI_LOGLEVEL_HPP

