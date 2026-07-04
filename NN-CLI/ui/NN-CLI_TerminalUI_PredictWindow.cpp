#include "NN-CLI_TerminalUI_PredictWindow.hpp"

namespace NN_CLI
{

  //===================================================================================================================//
  //-- Ctors / Dtors --//
  //===================================================================================================================//

  TerminalUI_PredictWindow::TerminalUI_PredictWindow()
    : TerminalUI_ResultsWindow("---- PREDICT ----", 1,
                               {{"Index", 6, TerminalUI_Table::Align::RIGHT},
                                {"Predicted Class", 16, TerminalUI_Table::Align::LEFT},
                                {"Confidence", 11, TerminalUI_Table::Align::RIGHT}},
                               {6, 16, 11})
  {
  }

  //===================================================================================================================//

  TerminalUI_PredictWindow::~TerminalUI_PredictWindow() {}

} // namespace NN_CLI
