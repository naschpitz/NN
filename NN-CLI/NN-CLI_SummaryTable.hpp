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
      // maxWidth=0 means auto-detect from terminal; positive maxWidth used when rendering
      // in a panel (e.g. config panel) and the terminal ioctl may not reflect the panel size.
      static std::vector<std::string> collect(const std::string& title, const std::vector<SummaryRow>& rows,
                                              ulong maxWidth = 0);

      // Variant with pre-computed column widths (both tables get identical column sizes).
      static std::vector<std::string> collect(const std::string& title, const std::vector<SummaryRow>& rows,
                                              ulong maxWidth, ulong keyW, ulong valueW);

      // Section descriptor for uniform-width multi-section tables.
      struct Section {
          std::string title;
          std::vector<SummaryRow> rows;
      };

      // Generate multiple sections with consistent column widths across all sections.
      static std::vector<std::string> collectSections(const std::vector<Section>& sections, ulong maxWidth);

      static std::string formatWithCommas(ulong value);
  };

} // namespace NN_CLI

#endif // NN_CLI_SUMMARYTABLE_HPP
