#ifndef NN_CLI_TERMINALUI_MODELINFOTABLE_HPP
#define NN_CLI_TERMINALUI_MODELINFOTABLE_HPP

#include "NN-CLI_TerminalUI_Table.hpp"

#include <string>
#include <vector>
#include <utility>

namespace NN_CLI
{

  // Specialized two-column table for displaying model configuration and
  // training hyperparameters as key-value pairs.
  //
  // Inherits from TerminalUI_Table and pre-configures two columns
  // ("Property" and "Value") with sensible defaults.  The convenience API
  // (addEntry / setEntries / clearEntries) lets callers populate data
  // without manually constructing Column descriptors or Row vectors.
  //
  // A centered title row (via setTitle) is intended to hold the model name
  // or section label.  The table renders via the base-class render() /
  // draw() methods.

  class TerminalUI_ModelInfoTable : public TerminalUI_Table
  {
    public:
      //-- Types --//

      // A single key-value pair describing one model property.
      using Entry = std::pair<std::string, std::string>;

      //-- Ctors --//

      TerminalUI_ModelInfoTable();

      // Construct with an explicit max rendering width.
      explicit TerminalUI_ModelInfoTable(int maxWidth);

      //-- Entries --//

      // Append a single key-value entry as a new row.
      void addEntry(const std::string& key, const std::string& value);

      // Replace all entries with the provided list.
      void setEntries(const std::vector<Entry>& entries);

      // Remove all entries.
      void clearEntries();
  };

} // namespace NN_CLI

#endif // NN_CLI_TERMINALUI_MODELINFOTABLE_HPP
