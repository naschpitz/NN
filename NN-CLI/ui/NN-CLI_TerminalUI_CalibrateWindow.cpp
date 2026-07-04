#include "NN-CLI_TerminalUI_CalibrateWindow.hpp"

namespace NN_CLI
{

  //===================================================================================================================//
  //-- Ctors / Dtors --//
  //===================================================================================================================//

  TerminalUI_CalibrateWindow::TerminalUI_CalibrateWindow()
    : TerminalUI_ResultsWindow(
        "---- CALIBRATE ----", 7,
        {{"Metric", 24, TerminalUI_Table::Align::LEFT}, {"Value", 20, TerminalUI_Table::Align::LEFT}}, {24, 20})
  {
  }

  //===================================================================================================================//

  TerminalUI_CalibrateWindow::~TerminalUI_CalibrateWindow() {}

} // namespace NN_CLI
