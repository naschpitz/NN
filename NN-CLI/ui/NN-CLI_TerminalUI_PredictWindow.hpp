#ifndef NN_CLI_TERMINALUI_PREDICTWINDOW_HPP
#define NN_CLI_TERMINALUI_PREDICTWINDOW_HPP

#include "NN-CLI_TerminalUI_ResultsWindow.hpp"

namespace NN_CLI
{

  // Specialized ResultsWindow for predict mode.  Inherits the full
  // five-panel layout, progress, scroll, and dismiss behavior from
  // TerminalUI_ResultsWindow; only fixes the title/color and the
  // results-table columns (Index / Predicted Class / Confidence).

  class TerminalUI_PredictWindow : public TerminalUI_ResultsWindow
  {
    public:
      //-- Ctors / Dtors --//

      TerminalUI_PredictWindow();

      ~TerminalUI_PredictWindow() override;
  };

} // namespace NN_CLI

#endif // NN_CLI_TERMINALUI_PREDICTWINDOW_HPP
