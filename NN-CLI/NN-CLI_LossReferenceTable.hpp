#ifndef NN_CLI_LOSSREFERENCETABLE_HPP
#define NN_CLI_LOSSREFERENCETABLE_HPP

#include <sys/types.h>

namespace NN_CLI
{

  using ulong = unsigned long;

  class LossReferenceTable
  {
    public:
      static void print(ulong numClasses);
  };

} // namespace NN_CLI

#endif // NN_CLI_LOSSREFERENCETABLE_HPP
