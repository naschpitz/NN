#include "test_helpers.hpp"

#include "NN-CLI_TerminalUI_TrainingWindow.hpp"

#include <string>
#include <vector>

// ncurses key constants used by the tests.  Defined as octal literals to
// match <curses.h> without actually including it (which defines a `timeout`
// macro that conflicts with Qt headers).
namespace TestKeys
{
  constexpr int kKeyUp   = 0403;  // KEY_UP
  constexpr int kKeyDown = 0402;  // KEY_DOWN
  constexpr int kKeyPgUp = 0523;  // KEY_PPAGE
  constexpr int kKeyPgDn = 0522;  // KEY_NPAGE
  constexpr int kKeyHome = 0406;  // KEY_HOME
  constexpr int kKeyEnd  = 0550;  // KEY_END
} // namespace TestKeys

//===================================================================================================================//

static void populateScrollablePanel(NN_CLI::TerminalUI_Panel* panel, int lineCount)
{
  // Give the panel enough size to have a positive content area.
  panel->resize(80, 10, 0, 0);

  std::vector<std::string> lines;
  lines.reserve(static_cast<ulong>(lineCount));

  for (int i = 0; i < lineCount; i++)
    lines.emplace_back("Line " + std::to_string(i));

  panel->setLines(lines);
  panel->setAutoScroll(false);
}

//===================================================================================================================//

void runTerminalUITests()
{
  std::cout << "=== Terminal UI Tests ===" << std::endl;

  //-- cycleActivePanel --//

  {
    NN_CLI::TerminalUI_TrainingWindow window;

    // Default active panel is MODEL_INFO (0).
    CHECK(window.getActivePanel() == NN_CLI::TerminalUI_TrainingWindow::MODEL_INFO,
          "Default active panel should be MODEL_INFO");

    // Tab should cycle: 0 -> 1.
    CHECK(window.cycleActivePanel('\t'), "Tab should be consumed");
    CHECK(window.getActivePanel() == NN_CLI::TerminalUI_TrainingWindow::EPOCHS,
          "After one Tab, active panel should be EPOCHS (1)");

    // Tab again: 1 -> 2.
    CHECK(window.cycleActivePanel('\t'), "Second Tab should be consumed");
    CHECK(window.getActivePanel() == NN_CLI::TerminalUI_TrainingWindow::TIMING,
          "After two Tabs, active panel should be TIMING (2)");

    // Tab wraps: 2 -> 0.
    CHECK(window.cycleActivePanel('\t'), "Third Tab should be consumed");
    CHECK(window.getActivePanel() == NN_CLI::TerminalUI_TrainingWindow::MODEL_INFO,
          "After three Tabs, active panel should wrap back to MODEL_INFO (0)");

    // Non-Tab keys should not be consumed.
    CHECK(!window.cycleActivePanel('a'), "Non-Tab key should not be consumed by cycleActivePanel");
    CHECK(!window.cycleActivePanel(TestKeys::kKeyUp), "KEY_UP should not be consumed by cycleActivePanel");
  }

  //-- scrollActivePanel --//

  {
    NN_CLI::TerminalUI_TrainingWindow window;

    // Populate the Model Info panel with scrollable content.
    populateScrollablePanel(window.getModelInfoPanel(), 50);

    window.setActivePanel(NN_CLI::TerminalUI_TrainingWindow::MODEL_INFO);

    int initialOffset = window.getModelInfoPanel()->scrollState().offset;
    CHECK(initialOffset == 0, "Initial scroll offset should be 0");

    // Scroll down on the active (Model Info) panel.
    CHECK(window.scrollActivePanel(TestKeys::kKeyDown), "KEY_DOWN should be consumed by scrollActivePanel");
    CHECK(window.getModelInfoPanel()->scrollState().offset == 1,
          "Scroll offset should increase to 1 after KEY_DOWN");

    // Scroll down again.
    CHECK(window.scrollActivePanel(TestKeys::kKeyDown), "Second KEY_DOWN should be consumed");
    CHECK(window.getModelInfoPanel()->scrollState().offset == 2,
          "Scroll offset should increase to 2 after second KEY_DOWN");

    // Scroll up.
    CHECK(window.scrollActivePanel(TestKeys::kKeyUp), "KEY_UP should be consumed");
    CHECK(window.getModelInfoPanel()->scrollState().offset == 1,
          "Scroll offset should decrease to 1 after KEY_UP");

    // Page down — offset should increase by content height.
    CHECK(window.scrollActivePanel(TestKeys::kKeyPgDn), "KEY_NPAGE should be consumed");
    int offsetAfterPageDn = window.getModelInfoPanel()->scrollState().offset;
    CHECK(offsetAfterPageDn > 1, "Page down should advance offset beyond 1");

    // Home — offset should be 0.
    CHECK(window.scrollActivePanel(TestKeys::kKeyHome), "KEY_HOME should be consumed");
    CHECK(window.getModelInfoPanel()->scrollState().offset == 0,
          "KEY_HOME should reset offset to 0");

    // Non-scroll key should not be consumed.
    CHECK(!window.scrollActivePanel('a'), "Non-scroll key should not be consumed by scrollActivePanel");
    CHECK(!window.scrollActivePanel('\t'), "Tab should not be consumed by scrollActivePanel");
  }

  //-- Scroll routing targets the active panel --//

  {
    NN_CLI::TerminalUI_TrainingWindow window;

    // Make all three panels scrollable (autoScroll disabled by populateScrollablePanel).
    populateScrollablePanel(window.getModelInfoPanel(), 50);
    populateScrollablePanel(window.getEpochsPanel(), 50);
    populateScrollablePanel(window.getTimingPanel(), 50);

    // Activate EPOCHS panel (index 1).
    window.setActivePanel(NN_CLI::TerminalUI_TrainingWindow::EPOCHS);

    // Scroll down — should affect EPOCHS only, not MODEL_INFO or TIMING.
    CHECK(window.scrollActivePanel(TestKeys::kKeyDown), "KEY_DOWN on EPOCHS panel should be consumed");
    CHECK(window.getEpochsPanel()->scrollState().offset == 1,
          "EPOCHS offset should be 1 after scroll");
    CHECK(window.getModelInfoPanel()->scrollState().offset == 0,
          "MODEL_INFO offset should remain 0 (not the active panel)");
    CHECK(window.getTimingPanel()->scrollState().offset == 0,
          "TIMING offset should remain 0 (not the active panel)");

    // Switch to TIMING panel (index 2) and scroll.
    window.setActivePanel(NN_CLI::TerminalUI_TrainingWindow::TIMING);
    CHECK(window.scrollActivePanel(TestKeys::kKeyEnd), "KEY_END on TIMING panel should be consumed");
    CHECK(window.getTimingPanel()->scrollState().offset > 0,
          "TIMING offset should be non-zero after KEY_END");
    CHECK(window.getModelInfoPanel()->scrollState().offset == 0,
          "MODEL_INFO offset should still be 0");
    CHECK(window.getEpochsPanel()->scrollState().offset == 1,
          "EPOCHS offset should still be 1 from earlier scroll");
  }

  //-- handleEvent delegates correctly --//

  {
    NN_CLI::TerminalUI_TrainingWindow window;
    populateScrollablePanel(window.getModelInfoPanel(), 50);
    populateScrollablePanel(window.getEpochsPanel(), 50);

    // handleEvent should delegate Tab to cycleActivePanel.
    CHECK(window.handleEvent('\t'), "handleEvent should consume Tab");
    CHECK(window.getActivePanel() == NN_CLI::TerminalUI_TrainingWindow::EPOCHS,
          "handleEvent Tab should cycle panel to EPOCHS");

    // handleEvent should delegate scroll keys to scrollActivePanel.
    CHECK(window.handleEvent(TestKeys::kKeyDown), "handleEvent should consume KEY_DOWN");
    CHECK(window.getEpochsPanel()->scrollState().offset == 1,
          "handleEvent KEY_DOWN should scroll active panel");

    // handleEvent should return false for unrecognized keys.
    CHECK(!window.handleEvent(-1), "handleEvent should not consume unknown key -1");
  }
}
