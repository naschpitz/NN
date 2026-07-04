#ifndef NN_CLI_TERMINALUI_CALIBRATEWINDOW_HPP
#define NN_CLI_TERMINALUI_CALIBRATEWINDOW_HPP

#include "NN-CLI_TerminalUI_ResultsWindow.hpp"

namespace NN_CLI
{

  // Specialized ResultsWindow for calibration mode.  Inherits the full
  // five-panel layout, progress, scroll, and dismiss behavior from
  // TerminalUI_ResultsWindow; only fixes the title/color and the
  // results-table columns (Metric / Value), which the
  // CalibrateController fills from the CalibrateResult payload
  // (free-energy threshold, ID/OOD counts, acceptance/rejection rates).

  class TerminalUI_CalibrateWindow : public TerminalUI_ResultsWindow
  {
    public:
      //-- Ctors / Dtors --//

      TerminalUI_CalibrateWindow();

      ~TerminalUI_CalibrateWindow() override;
  };

} // namespace NN_CLI

#endif // NN_CLI_TERMINALUI_CALIBRATEWINDOW_HPP
