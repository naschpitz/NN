#ifndef NN_CLI_SUMMARYTABLE_HPP
#define NN_CLI_SUMMARYTABLE_HPP

#include <string>
#include <vector>

namespace NN_CLI
{

  using ulong = unsigned long;

  // Shared table rendering for all summary types (train, test, predict).
  struct SummaryRow {
      std::string key;
      std::string value; // Empty key = section separator
  };

  class SummaryTable
  {
    public:
      // Print an auto-sized ASCII table with a centered title.
      static void print(const std::string& title, const std::vector<SummaryRow>& rows);

      // Format a number with commas: 1234567 → "1,234,567"
      static std::string formatWithCommas(ulong value);
  };

} // namespace NN_CLI

#endif // NN_CLI_SUMMARYTABLE_HPP
