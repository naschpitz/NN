#include "test_helpers.hpp"

#include "NN-CLI_TerminalUI_TrainWindow.hpp"
#include "NN-CLI_TerminalUI_PredictWindow.hpp"

#include <cstdio>
#include <string>
#include <vector>

// ncurses key constants used by the tests.  Defined as octal literals to
// match <curses.h> without actually including it (which defines a `timeout`
// macro that conflicts with Qt headers).
namespace TestKeys
{
  constexpr int kKeyUp = 0403; // KEY_UP
  constexpr int kKeyDown = 0402; // KEY_DOWN
  constexpr int kKeyPgUp = 0523; // KEY_PPAGE
  constexpr int kKeyPgDn = 0522; // KEY_NPAGE
  constexpr int kKeyHome = 0406; // KEY_HOME
  constexpr int kKeyEnd = 0550; // KEY_END
} // namespace TestKeys

//===================================================================================================================//

// Count non-continuation bytes (= Unicode code points = visual columns for
// the narrow characters used in the table).  Mirrors the logic in
// TerminalUI_Table.cpp's anonymous-namespace getVisualWidth().
static int getVisualWidth(const std::string& text)
{
  int width = 0;

  for (unsigned char c : text) {
    if ((c & 0xC0) != 0x80)
      width++;
  }

  return width;
}

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
    NN_CLI::TerminalUI_TrainWindow window;

    // Default active panel is MODEL_INFO (0).
    CHECK(window.getActivePanel() == NN_CLI::TerminalUI_TrainWindow::MODEL_INFO,
          "Default active panel should be MODEL_INFO");

    // Tab should cycle: 0 -> 1.
    CHECK(window.cycleActivePanel('\t'), "Tab should be consumed");
    CHECK(window.getActivePanel() == NN_CLI::TerminalUI_TrainWindow::EPOCHS,
          "After one Tab, active panel should be EPOCHS (1)");

    // Tab again: 1 -> 2.
    CHECK(window.cycleActivePanel('\t'), "Second Tab should be consumed");
    CHECK(window.getActivePanel() == NN_CLI::TerminalUI_TrainWindow::TIMING,
          "After two Tabs, active panel should be TIMING (2)");

    // Tab wraps: 2 -> 0.
    CHECK(window.cycleActivePanel('\t'), "Third Tab should be consumed");
    CHECK(window.getActivePanel() == NN_CLI::TerminalUI_TrainWindow::MODEL_INFO,
          "After three Tabs, active panel should wrap back to MODEL_INFO (0)");

    // Non-Tab keys should not be consumed.
    CHECK(!window.cycleActivePanel('a'), "Non-Tab key should not be consumed by cycleActivePanel");
    CHECK(!window.cycleActivePanel(TestKeys::kKeyUp), "KEY_UP should not be consumed by cycleActivePanel");
  }

  std::cout << "PASS: TrainWindow: cycleActivePanel" << std::endl;

  //-- scrollActivePanel --//

  {
    NN_CLI::TerminalUI_TrainWindow window;

    // Populate the Model Info panel with scrollable content.
    populateScrollablePanel(window.getModelInfoPanel(), 50);

    window.setActivePanel(NN_CLI::TerminalUI_TrainWindow::MODEL_INFO);

    int initialOffset = window.getModelInfoPanel()->scrollState().offset;
    CHECK(initialOffset == 0, "Initial scroll offset should be 0");

    // Scroll down on the active (Model Info) panel.
    CHECK(window.scrollActivePanel(TestKeys::kKeyDown), "KEY_DOWN should be consumed by scrollActivePanel");
    CHECK(window.getModelInfoPanel()->scrollState().offset == 1, "Scroll offset should increase to 1 after KEY_DOWN");

    // Scroll down again.
    CHECK(window.scrollActivePanel(TestKeys::kKeyDown), "Second KEY_DOWN should be consumed");
    CHECK(window.getModelInfoPanel()->scrollState().offset == 2,
          "Scroll offset should increase to 2 after second KEY_DOWN");

    // Scroll up.
    CHECK(window.scrollActivePanel(TestKeys::kKeyUp), "KEY_UP should be consumed");
    CHECK(window.getModelInfoPanel()->scrollState().offset == 1, "Scroll offset should decrease to 1 after KEY_UP");

    // Page down — offset should increase by content height.
    CHECK(window.scrollActivePanel(TestKeys::kKeyPgDn), "KEY_NPAGE should be consumed");
    int offsetAfterPageDn = window.getModelInfoPanel()->scrollState().offset;
    CHECK(offsetAfterPageDn > 1, "Page down should advance offset beyond 1");

    // Home — offset should be 0.
    CHECK(window.scrollActivePanel(TestKeys::kKeyHome), "KEY_HOME should be consumed");
    CHECK(window.getModelInfoPanel()->scrollState().offset == 0, "KEY_HOME should reset offset to 0");

    // Non-scroll key should not be consumed.
    CHECK(!window.scrollActivePanel('a'), "Non-scroll key should not be consumed by scrollActivePanel");
    CHECK(!window.scrollActivePanel('\t'), "Tab should not be consumed by scrollActivePanel");
  }

  std::cout << "PASS: TrainWindow: scrollActivePanel" << std::endl;

  //-- Scroll routing targets the active panel --//

  {
    NN_CLI::TerminalUI_TrainWindow window;

    // Make all three panels scrollable (autoScroll disabled by populateScrollablePanel).
    populateScrollablePanel(window.getModelInfoPanel(), 50);
    populateScrollablePanel(window.getEpochsPanel(), 50);
    populateScrollablePanel(window.getTimingPanel(), 50);

    // Activate EPOCHS panel (index 1).
    window.setActivePanel(NN_CLI::TerminalUI_TrainWindow::EPOCHS);

    // Scroll down — should affect EPOCHS only, not MODEL_INFO or TIMING.
    CHECK(window.scrollActivePanel(TestKeys::kKeyDown), "KEY_DOWN on EPOCHS panel should be consumed");
    CHECK(window.getEpochsPanel()->scrollState().offset == 1, "EPOCHS offset should be 1 after scroll");
    CHECK(window.getModelInfoPanel()->scrollState().offset == 0,
          "MODEL_INFO offset should remain 0 (not the active panel)");
    CHECK(window.getTimingPanel()->scrollState().offset == 0, "TIMING offset should remain 0 (not the active panel)");

    // Switch to TIMING panel (index 2) and scroll.
    window.setActivePanel(NN_CLI::TerminalUI_TrainWindow::TIMING);
    CHECK(window.scrollActivePanel(TestKeys::kKeyEnd), "KEY_END on TIMING panel should be consumed");
    CHECK(window.getTimingPanel()->scrollState().offset > 0, "TIMING offset should be non-zero after KEY_END");
    CHECK(window.getModelInfoPanel()->scrollState().offset == 0, "MODEL_INFO offset should still be 0");
    CHECK(window.getEpochsPanel()->scrollState().offset == 1, "EPOCHS offset should still be 1 from earlier scroll");
  }

  std::cout << "PASS: TrainWindow: scroll routing targets active panel" << std::endl;

  //-- handleEvent delegates correctly --//

  {
    NN_CLI::TerminalUI_TrainWindow window;
    populateScrollablePanel(window.getModelInfoPanel(), 50);
    populateScrollablePanel(window.getEpochsPanel(), 50);

    // handleEvent should delegate Tab to cycleActivePanel.
    CHECK(window.handleEvent('\t'), "handleEvent should consume Tab");
    CHECK(window.getActivePanel() == NN_CLI::TerminalUI_TrainWindow::EPOCHS,
          "handleEvent Tab should cycle panel to EPOCHS");

    // handleEvent should delegate scroll keys to scrollActivePanel.
    CHECK(window.handleEvent(TestKeys::kKeyDown), "handleEvent should consume KEY_DOWN");
    CHECK(window.getEpochsPanel()->scrollState().offset == 1, "handleEvent KEY_DOWN should scroll active panel");

    // handleEvent should return false for unrecognized keys.
    CHECK(!window.handleEvent(-1), "handleEvent should not consume unknown key -1");
  }

  std::cout << "PASS: TrainWindow: handleEvent delegates correctly" << std::endl;

  //-- Table rendering with multi-byte UTF-8 preserves right border --//

  {
    // Regression: the rightmost "|" divider of the last column was truncated
    // by mvaddnstr because line.size() (byte count) exceeded the visual-
    // column limit (maxW) when the row contained multi-byte UTF-8 characters
    // like the checkmark (U+2713, 3 bytes, 1 visual column).

    NN_CLI::TerminalUI_Table table;
    table.setColumns({
      {"Epoch", 5, NN_CLI::TerminalUI_Table::Align::RIGHT},
      {"Loss", 8, NN_CLI::TerminalUI_Table::Align::RIGHT},
      {"Validation Loss", 15, NN_CLI::TerminalUI_Table::Align::RIGHT},
      {"Best", 4, NN_CLI::TerminalUI_Table::Align::LEFT},
      {"Completed At", 19, NN_CLI::TerminalUI_Table::Align::LEFT},
    });

    // Use the same maxWidth a typical panel would provide.
    const int maxWidth = 76;
    table.setMaxWidth(maxWidth);

    // Row WITHOUT multi-byte chars — pure ASCII baseline.
    table.addRow({"1", "2.345678", "1.987654", "", "Jun 11, 2026 14:30"});

    // Row WITH checkmark — multi-byte UTF-8 in the "Best" column.
    // The checkmark U+2713 is encoded as 3 bytes in UTF-8 (E2 9C 93) but
    // occupies only 1 visual column.
    table.addRow({"2", "1.234567", "0.876543", "✓", "Jun 11, 2026 14:35"});

    auto lines = table.render();

    // The table should produce at least separator + header + separator +
    // 2 data rows + separator = 6 lines.
    CHECK(static_cast<int>(lines.size()) >= 6, "Epoch table with 2 rows should produce at least 6 lines");

    // Data rows and header must end with the right border "|".
    // Separator lines end with "+" and are checked separately.
    for (int i = 0; i < static_cast<int>(lines.size()); i++) {
      char expectedBorder = '|';
      // Lines 0, 2, 5 (for 2 data rows) are separators ending with '+'.
      // All other lines are data/header rows ending with '|'.
      bool isSeparator = (lines[i].find('+') == 0);

      if (isSeparator) {
        // Separators are of the form "+---+---+..."
        CHECK(!lines[i].empty() && lines[i].back() == '+',
              "Separator line " + std::to_string(i) + " must end with '+'");
      } else {
        CHECK(!lines[i].empty() && lines[i].back() == '|',
              "Data/header line " + std::to_string(i) + " must end with '|'");
      }
    }

    // Find the data rows (lines[3] and [4] for the 2 data rows).
    // Line layout: [0]=sep, [1]=header, [2]=sep, [3]=row1, [4]=row2, [5]=sep
    const std::string& asciiRow = lines[3];
    const std::string& utf8Row = lines[4];

    // ASCII row: byte count == visual width.
    int asciiVisual = getVisualWidth(asciiRow);
    CHECK(static_cast<int>(asciiRow.size()) == asciiVisual, "ASCII row byte count must equal visual width");
    CHECK(asciiVisual == maxWidth, "ASCII row visual width must equal maxWidth (" + std::to_string(maxWidth) + ")");

    // UTF-8 row (with checkmark): byte count > visual width.
    int utf8Visual = getVisualWidth(utf8Row);
    CHECK(static_cast<int>(utf8Row.size()) > utf8Visual,
          "Row with checkmark must have byte count > visual width (multi-byte char)");
    CHECK(utf8Visual == maxWidth, "UTF-8 row visual width must equal maxWidth (" + std::to_string(maxWidth) + ")");

    // The critical invariant: the byte count exceeds maxWidth, so a naive
    // min(line.size(), maxW) would truncate the string.
    int utf8Bytes = static_cast<int>(utf8Row.size());
    int extraBytes = utf8Bytes - maxWidth;

    CHECK(extraBytes > 0, "Row with checkmark must have byte count > maxWidth");

    // With one checkmark (3 bytes, 1 col) the extra bytes should be exactly 2.
    CHECK(utf8Bytes == maxWidth + 2, "Row with one checkmark must have byte count = maxWidth + 2 (got " +
                                       std::to_string(utf8Bytes) + ", expected " + std::to_string(maxWidth + 2) + ")");

    // Simulate the OLD broken truncation: min(bytes, maxW) would give maxW
    // bytes, cutting off the trailing " |".
    int oldPrintLen = std::min(utf8Bytes, maxWidth);
    CHECK(oldPrintLen < utf8Bytes, "Old truncation min(bytes, maxW) must be less than full byte count");

    // The last 2 bytes of the UTF-8 row are " |" — the right border.
    CHECK(utf8Row[utf8Bytes - 2] == ' ' && utf8Row[utf8Bytes - 1] == '|',
          "Last 2 bytes of UTF-8 row must be ' |' (space + right border)");

    // -- Simulate both truncation strategies and prove the fix -- //

    // OLD truncation (byte-based): min(bytes, maxW).
    // This cuts off the last 2 bytes (" |"), losing the right border.
    int oldTruncLen = std::min(utf8Bytes, maxWidth);
    std::string oldTruncated = utf8Row.substr(0, static_cast<std::string::size_type>(oldTruncLen));
    CHECK(oldTruncated.back() != '|', "OLD byte-based truncation must lose the right border '|'");

    // NEW truncation (visual-width-based): walk UTF-8 sequences up to maxW cols.
    // This preserves the full visual width including the right border '|'.
    int newBytes = 0;
    int cols = 0;

    while (newBytes < utf8Bytes && cols < maxWidth) {
      unsigned char c = static_cast<unsigned char>(utf8Row[newBytes]);
      int charBytes = 1;

      if ((c & 0xE0) == 0xC0)
        charBytes = 2;
      else if ((c & 0xF0) == 0xE0)
        charBytes = 3;
      else if ((c & 0xF8) == 0xF0)
        charBytes = 4;

      if (newBytes + charBytes > utf8Bytes)
        break;
      cols++;
      newBytes += charBytes;
    }

    std::string newTruncated = utf8Row.substr(0, static_cast<std::string::size_type>(newBytes));
    CHECK(newTruncated.back() == '|', "NEW visual-width truncation must preserve the right border '|'");
    CHECK(static_cast<int>(newTruncated.size()) == utf8Bytes,
          "NEW truncation must include all bytes (nothing cut off)");
  }

  std::cout << "PASS: Table rendering with multi-byte UTF-8 preserves right border" << std::endl;

  //-- PredictWindow: default active panel --//

  {
    NN_CLI::TerminalUI_PredictWindow pw;
    CHECK(pw.getActivePanel() == NN_CLI::TerminalUI_PredictWindow::MODEL_INFO,
          "PredictWindow default active panel should be MODEL_INFO");
  }

  std::cout << "PASS: PredictWindow: default active panel" << std::endl;

  //-- PredictWindow: Tab cycles panels --//

  {
    NN_CLI::TerminalUI_PredictWindow pw;

    // Tab cycles: MODEL_INFO(0) -> TIMING(1) -> EPOCH_HISTORY(2) -> RESULTS(3) -> PROGRESS(4) -> MODEL_INFO(0)
    CHECK(pw.cycleActivePanel('\t'), "Tab should be consumed");
    CHECK(pw.getActivePanel() == NN_CLI::TerminalUI_PredictWindow::TIMING,
          "After one Tab, active panel should be TIMING (1)");

    pw.cycleActivePanel('\t');
    CHECK(pw.getActivePanel() == NN_CLI::TerminalUI_PredictWindow::EPOCH_HISTORY,
          "After two Tabs, active panel should be EPOCH_HISTORY (2)");

    pw.cycleActivePanel('\t');
    CHECK(pw.getActivePanel() == NN_CLI::TerminalUI_PredictWindow::RESULTS,
          "After three Tabs, active panel should be RESULTS (3)");

    pw.cycleActivePanel('\t');
    CHECK(pw.getActivePanel() == NN_CLI::TerminalUI_PredictWindow::PROGRESS,
          "After four Tabs, active panel should be PROGRESS (4)");

    pw.cycleActivePanel('\t');
    CHECK(pw.getActivePanel() == NN_CLI::TerminalUI_PredictWindow::MODEL_INFO,
          "After five Tabs, active panel should wrap back to MODEL_INFO (0)");

    // Non-Tab keys should not be consumed.
    CHECK(!pw.cycleActivePanel('a'), "Non-Tab key should not be consumed by cycleActivePanel");
    CHECK(!pw.cycleActivePanel(TestKeys::kKeyUp), "KEY_UP should not be consumed by cycleActivePanel");
  }

  std::cout << "PASS: PredictWindow: Tab cycles panels" << std::endl;

  //-- PredictWindow: scroll routes to active panel --//

  {
    NN_CLI::TerminalUI_PredictWindow pw;

    // Populate the Model Info panel with scrollable content.
    populateScrollablePanel(pw.getModelInfoPanel(), 50);

    pw.setActivePanel(NN_CLI::TerminalUI_PredictWindow::MODEL_INFO);

    int initialOffset = pw.getModelInfoPanel()->scrollState().offset;
    CHECK(initialOffset == 0, "Initial scroll offset should be 0");

    // Scroll down on the active (Model Info) panel.
    CHECK(pw.scrollActivePanel(TestKeys::kKeyDown), "KEY_DOWN should be consumed by scrollActivePanel");
    CHECK(pw.getModelInfoPanel()->scrollState().offset == 1, "Scroll offset should increase to 1 after KEY_DOWN");

    // Scroll down again.
    CHECK(pw.scrollActivePanel(TestKeys::kKeyDown), "Second KEY_DOWN should be consumed");
    CHECK(pw.getModelInfoPanel()->scrollState().offset == 2,
          "Scroll offset should increase to 2 after second KEY_DOWN");

    // Scroll up.
    CHECK(pw.scrollActivePanel(TestKeys::kKeyUp), "KEY_UP should be consumed");
    CHECK(pw.getModelInfoPanel()->scrollState().offset == 1, "Scroll offset should decrease to 1 after KEY_UP");

    // Home — offset should be 0.
    CHECK(pw.scrollActivePanel(TestKeys::kKeyHome), "KEY_HOME should be consumed");
    CHECK(pw.getModelInfoPanel()->scrollState().offset == 0, "KEY_HOME should reset offset to 0");

    // Non-scroll key should not be consumed.
    CHECK(!pw.scrollActivePanel('a'), "Non-scroll key should not be consumed by scrollActivePanel");
    CHECK(!pw.scrollActivePanel('\t'), "Tab should not be consumed by scrollActivePanel");
  }

  std::cout << "PASS: PredictWindow: scroll routes to active panel" << std::endl;

  //-- PredictWindow: handleEvent delegates correctly --//

  {
    NN_CLI::TerminalUI_PredictWindow pw;
    populateScrollablePanel(pw.getModelInfoPanel(), 50);
    populateScrollablePanel(pw.getTimingPanel(), 50);

    // handleEvent should delegate Tab to cycleActivePanel.
    CHECK(pw.handleEvent('\t'), "handleEvent should consume Tab");
    CHECK(pw.getActivePanel() == NN_CLI::TerminalUI_PredictWindow::TIMING,
          "handleEvent Tab should cycle panel to TIMING");

    // handleEvent should delegate scroll keys to scrollActivePanel.
    // After Tab the active panel is TIMING, so the scroll must land on it
    // (mirrors the TrainWindow handleEvent test).
    CHECK(pw.handleEvent(TestKeys::kKeyDown), "handleEvent should consume KEY_DOWN");
    CHECK(pw.getTimingPanel()->scrollState().offset == 1, "handleEvent KEY_DOWN should scroll active panel");

    // handleEvent should return false for unrecognized keys.
    CHECK(!pw.handleEvent(-1), "handleEvent should not consume unknown key -1");
  }

  std::cout << "PASS: PredictWindow: handleEvent delegates correctly" << std::endl;

  //-- PredictWindow: dismiss on 'q' --//

  {
    NN_CLI::TerminalUI_PredictWindow pw;
    CHECK(pw.handleEvent('q'), "handleEvent 'q' should set dismissed flag and return true");
    // waitForDismiss should return immediately since dismissed_ is already true.
    pw.waitForDismiss();
    CHECK(true, "waitForDismiss returned immediately after 'q' dismiss");
  }

  std::cout << "PASS: PredictWindow: dismiss on 'q'" << std::endl;

  //-- PredictWindow: results table renders header + data rows --//

  {
    NN_CLI::TerminalUI_PredictWindow pw;

    // Resize the results panel so contentWidth() returns a usable width.
    pw.getResultsPanel()->resize(80, 10, 0, 0);

    pw.setResultsColumns({"Index", "Predicted Class", "Confidence"});
    pw.addResultRow({"0", "3", "95%"});
    pw.addResultRow({"1", "7", "87%"});
    pw.refreshResultsContent();

    auto lines = pw.getResultsPanel()->getLines();

    // Expected layout: separator + header + separator + row1 + row2 + separator = 6 lines
    CHECK(static_cast<int>(lines.size()) >= 6,
          "Results table with 2 data rows should produce at least 6 lines (sep+hdr+sep+row1+row2+sep)");
  }

  std::cout << "PASS: PredictWindow: results table renders header + data rows" << std::endl;

  //-- PredictWindow: 5 child panels --//

  {
    NN_CLI::TerminalUI_PredictWindow pw;
    CHECK(pw.childCount() == 5, "PredictWindow should have 5 child panels");
  }

  std::cout << "PASS: PredictWindow: 5 child panels" << std::endl;
}
