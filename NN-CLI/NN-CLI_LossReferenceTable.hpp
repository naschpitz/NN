#ifndef NN_CLI_LOSSREFERENCETABLE_HPP
#define NN_CLI_LOSSREFERENCETABLE_HPP

#include "NN-CLI_SummaryTable.hpp"
#include "NN-CLI_Types.hpp"

#include <string>
#include <vector>

namespace NN_CLI
{

  class LossReferenceTable
  {
    public:
      static std::vector<std::string> collect(ulong numClasses, ulong maxWidth = 0);
      static std::vector<SummaryRow> collectRows(ulong numClasses);
  };

} // namespace NN_CLI

#endif // NN_CLI_LOSSREFERENCETABLE_HPP
