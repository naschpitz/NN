#include "NN-CLI_TerminalUI_ModelInfoTable.hpp"

namespace NN_CLI
{

  //===================================================================================================================//
  //-- Ctors --//
  //===================================================================================================================//

  TerminalUI_ModelInfoTable::TerminalUI_ModelInfoTable()
  {
    this->setColumns({
      {"Property", 18, TerminalUI_Table::Align::LEFT},
      {"Value", 30, TerminalUI_Table::Align::LEFT},
    });
  }

  //===================================================================================================================//

  TerminalUI_ModelInfoTable::TerminalUI_ModelInfoTable(int maxWidth) : TerminalUI_Table(maxWidth)
  {
    this->setColumns({
      {"Property", 18, TerminalUI_Table::Align::LEFT},
      {"Value", 30, TerminalUI_Table::Align::LEFT},
    });
  }

  //===================================================================================================================//
  //-- Entries --//
  //===================================================================================================================//

  void TerminalUI_ModelInfoTable::addEntry(const std::string& key, const std::string& value)
  {
    this->addRow({key, value});
  }

  //===================================================================================================================//

  void TerminalUI_ModelInfoTable::setEntries(const std::vector<Entry>& entries)
  {
    this->clearRows();

    for (const auto& entry : entries)
      this->addRow({entry.first, entry.second});
  }

  //===================================================================================================================//

  void TerminalUI_ModelInfoTable::clearEntries()
  {
    this->clearRows();
  }

} // namespace NN_CLI
