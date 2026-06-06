#ifndef NN_CLI_LOSSREFERENCETABLE_HPP
#define NN_CLI_LOSSREFERENCETABLE_HPP

#include "NN-CLI_SummaryTable.hpp"

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
      static std::vector<SummaryRow> collectRows(ulong numClasses);
  };

} // namespace NN_CLI

#endif // NN_CLI_LOSSREFERENCETABLE_HPP
