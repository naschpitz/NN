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
      static void print(const std::string& title, const std::vector<SummaryRow>& rows);

      // Return table lines without printing (for rendering in ncurses).
      static std::vector<std::string> collect(const std::string& title, const std::vector<SummaryRow>& rows);

      static std::string formatWithCommas(ulong value);
  };

} // namespace NN_CLI

#endif // NN_CLI_SUMMARYTABLE_HPP
