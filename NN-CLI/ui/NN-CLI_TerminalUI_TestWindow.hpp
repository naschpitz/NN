#ifndef NN_CLI_TERMINALUI_TESTWINDOW_HPP
#define NN_CLI_TERMINALUI_TESTWINDOW_HPP

#include "NN-CLI_TerminalUI_ResultsWindow.hpp"

namespace NN_CLI
{

  // Specialized ResultsWindow for test mode.  Inherits the full
  // five-panel layout, progress, scroll, and dismiss behavior from
  // TerminalUI_ResultsWindow; only fixes the title/color and the
  // results-table columns (Class / Precision / Recall / F1 / Support),
  // which the TestController fills from Common::TestResult's per-class
  // confusion-matrix metrics.

  class TerminalUI_TestWindow : public TerminalUI_ResultsWindow
  {
    public:
      //-- Ctors / Dtors --//

      TerminalUI_TestWindow();

      ~TerminalUI_TestWindow() override;
  };

} // namespace NN_CLI

#endif // NN_CLI_TERMINALUI_TESTWINDOW_HPP
