#include "NN-CLI_TerminalUI_TestWindow.hpp"

namespace NN_CLI
{

  //===================================================================================================================//
  //-- Ctors / Dtors --//
  //===================================================================================================================//

  TerminalUI_TestWindow::TerminalUI_TestWindow()
    : TerminalUI_ResultsWindow("---- TEST ----", 5,
                               {{"Class", 8, TerminalUI_Table::Align::RIGHT},
                                {"Precision", 10, TerminalUI_Table::Align::RIGHT},
                                {"Recall", 10, TerminalUI_Table::Align::RIGHT},
                                {"F1", 10, TerminalUI_Table::Align::RIGHT},
                                {"Support", 10, TerminalUI_Table::Align::RIGHT}},
                               {8, 10, 10, 10, 10})
  {
  }

  //===================================================================================================================//

  TerminalUI_TestWindow::~TerminalUI_TestWindow() {}

} // namespace NN_CLI
