#ifndef NN_CLI_LOSSREFERENCETABLE_HPP
#define NN_CLI_LOSSREFERENCETABLE_HPP

#include <string>
#include <vector>

#include <sys/types.h>

namespace NN_CLI
{

  using ulong = unsigned long;

  class LossReferenceTable
  {
    public:
      static std::vector<std::string> collect(ulong numClasses, ulong maxWidth = 0);
  };

} // namespace NN_CLI

#endif // NN_CLI_LOSSREFERENCETABLE_HPP
