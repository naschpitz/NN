#include "NN-CLI_TerminalUI.hpp"

#include <algorithm>
#include <clocale>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <curses.h>

namespace NN_CLI
{

  namespace
  {
    TerminalUI* g_activeUI = nullptr;

    void sigwinchHandler(int)
    {
      if (g_activeUI)
        g_activeUI->requestResize();
    }

    void sigtermHandler(int sig)
    {
      if (g_activeUI) {
        g_activeUI->shutdown();
        std::signal(sig, SIG_DFL);
        std::raise(sig);
      }
    }
  } // namespace

  //===================================================================================================================//
  //-- Ctors / Dtors --//
  //===================================================================================================================//

  TerminalUI::TerminalUI()
    : trainingPanel(0, 0, 0, 0, "Training", 2)
    , epochsPanel(0, 0, 0, 0, "Epochs", 2)
    , configPanel(0, 0, 0, 0, "Configuration", 2)
    , timingPanel(0, 0, 0, 0, "Timing", 2)
  {
    this->epochsPanel.setAutoScroll(true);

    this->epochsTable.setColumns({
      {"Epoch",        5,  TerminalUI_Table::Align::RIGHT},
      {"Loss",         8,  TerminalUI_Table::Align::RIGHT},
      {"Val Loss",     8,  TerminalUI_Table::Align::RIGHT},
      {"Best",         4,  TerminalUI_Table::Align::LEFT},
      {"Completed At", 19, TerminalUI_Table::Align::LEFT},
    });
  }

  //===================================================================================================================//

  TerminalUI::~TerminalUI()
  {
    this->shutdown();
  }

  //===================================================================================================================//
  //-- Lifecycle --//
  //===================================================================================================================//

  bool TerminalUI::init()
  {
    if (this->initialized)
      return true;

    ::setlocale(LC_ALL, "");

    FILE* tty = ::fopen("/dev/tty", "r+");

    if (!tty)
      return false;

    SCREEN* screen = newterm(nullptr, tty, tty);

    if (!screen) {
      ::fclose(tty);
      return false;
    }

    set_term(screen);

    cbreak();
    noecho();
    curs_set(0);
    keypad(stdscr, TRUE);
    nodelay(stdscr, TRUE);

    if (has_colors()) {
      start_color();
      use_default_colors();

      init_pair(1, COLOR_GREEN, -1);
      init_pair(2, COLOR_CYAN, -1);
      init_pair(3, COLOR_YELLOW, -1);
      init_pair(4, COLOR_RED, -1);
      init_pair(5, COLOR_WHITE, -1);
      init_pair(6, COLOR_GREEN, -1);
      init_pair(7, COLOR_MAGENTA, -1);
      init_pair(8, COLOR_BLUE, -1);
    }

    this->layout();

    g_activeUI = this;
    std::signal(SIGWINCH, sigwinchHandler);
    std::signal(SIGINT, sigtermHandler);
    std::signal(SIGTERM, sigtermHandler);

    this->initialized = true;
    return true;
  }

  //===================================================================================================================//

  void TerminalUI::shutdown()
  {
    if (!this->initialized)
      return;

    this->initialized = false;
    g_activeUI = nullptr;
    std::signal(SIGWINCH, SIG_DFL);
    std::signal(SIGINT, SIG_DFL);
    std::signal(SIGTERM, SIG_DFL);

    if (this->loadingWin) {
      delwin(this->loadingWin);
      this->loadingWin = nullptr;
    }

    if (this->progressWin) {
      delwin(this->progressWin);
      this->progressWin = nullptr;
    }


    endwin();
  }

  //===================================================================================================================//
  //-- Layout --//
  //===================================================================================================================//

  void TerminalUI::layout()
  {
    endwin();
    refresh();
    clear();

    this->rows = getmaxy(stdscr);
    this->cols = getmaxx(stdscr);

    int minTimingW = 20;
    int minLeftW = 68;

    if (this->cols < minLeftW + minTimingW) {
      this->leftWidth = this->cols;
      this->timingWidth = 0;
    } else {
      int idealTimingW = this->cols * 35 / 100;
      this->timingWidth = std::max(minTimingW, std::min(this->cols - minLeftW, idealTimingW));
      this->leftWidth = this->cols - this->timingWidth;
    }

    int screenRows = this->rows - 1;
    int trainingH = 5;
    int remaining = std::max(8, screenRows - trainingH);
    int configH = std::max(3, std::min(remaining - 5, remaining * 55 / 100));
    int epochsH = std::max(3, remaining - configH);

    int trainingY = 0;
    int epochsY = trainingY + trainingH;
    int configY = epochsY + epochsH;

    //-- Resize panels --//

    this->trainingPanel.resize(trainingY, 0, trainingH, this->leftWidth);
    this->epochsPanel.resize(epochsY, 0, epochsH, this->leftWidth);
    this->configPanel.resize(configY, 0, configH, this->leftWidth);

    if (this->timingWidth > 0)
      this->timingPanel.resize(0, this->leftWidth, screenRows, this->timingWidth);

    //-- Recreate overlay sub-windows --//

    if (this->progressWin) {
      delwin(this->progressWin);
      this->progressWin = nullptr;
    }

    if (this->loadingWin) {
      delwin(this->loadingWin);
      this->loadingWin = nullptr;
    }


    int loadW = std::max(1, this->leftWidth - 4);

    this->loadingWin = newwin(1, loadW, trainingY + 1, 2);
    this->progressWin = newwin(2, loadW, trainingY + 2, 2);

    if (this->loadingWin) {
      werase(this->loadingWin);
      touchwin(this->loadingWin);
    }

    if (this->progressWin) {
      werase(this->progressWin);
      touchwin(this->progressWin);
    }


    if (this->resizeCallback)
      this->resizeCallback();
  }

  //===================================================================================================================//
  //-- Drawing --//
  //===================================================================================================================//

  void TerminalUI::drawAllPanels()
  {
    touchwin(stdscr);
    erase();

    // Update active panel colors: active=YELLOW(3), inactive=CYAN(2).
    this->trainingPanel.setColorPair(2);
    this->configPanel.setColorPair(this->activePanel == 0 ? 3 : 2);
    this->epochsPanel.setColorPair(this->activePanel == 1 ? 3 : 2);
    this->timingPanel.setColorPair(this->activePanel == 2 ? 3 : 2);

    // Draw all frames first (so content can overlap frame edges if needed).
    this->trainingPanel.drawFrame();
    this->epochsPanel.drawFrame();
    this->configPanel.drawFrame();

    if (this->timingWidth > 0)
      this->timingPanel.drawFrame();

    // Draw content and scrollbars for scrollable panels.
    this->epochsPanel.drawContent();
    this->epochsPanel.drawScrollbar();

    this->configPanel.drawContent();
    this->configPanel.drawScrollbar();

    if (this->timingWidth > 0) {
      this->timingPanel.drawContent();
      this->timingPanel.drawScrollbar();
    }

    // Help bar at the bottom of the screen.
    int helpY = this->rows - 1;
    mvaddch(helpY, 0, ACS_LLCORNER);
    mvhline(helpY, 1, ACS_HLINE, this->cols - 2);
    mvaddch(helpY, this->cols - 1, ACS_LRCORNER);

    attron(COLOR_PAIR(2) | A_BOLD);
    mvaddstr(helpY, 3, "Tab: select panel  jk/arrows: scroll  PgUp/PgDn: page  Home/End: jump");
    attroff(COLOR_PAIR(2) | A_BOLD);
  }

  //===================================================================================================================//

  void TerminalUI::present(bool runOverlay, bool touchSub)
  {
    if (this->epochLinesDirty) {
      this->renderEpochContent();
      this->epochLinesDirty = false;
    }

    if (this->configLinesDirty) {
      this->renderConfigContent();
      this->configLinesDirty = false;
    }

    this->drawAllPanels();
    wnoutrefresh(stdscr);

    // The loading bar (and any overlay) lives in its own window; repaint it on top of the panels
    // after layout() recreated/erased the windows.
    if (runOverlay && this->overlayCallback)
      this->overlayCallback();

    // drawAllPanels() blanks the stdscr cells beneath the sub-windows, so when a sub-window's own
    // content is unchanged we must touch it to force a full re-copy on top.
    if (this->loadingWin) {
      if (touchSub)
        touchwin(this->loadingWin);

      wnoutrefresh(this->loadingWin);
    }

    if (this->progressWin) {
      if (touchSub)
        touchwin(this->progressWin);

      wnoutrefresh(this->progressWin);
    }

    doupdate();
  }

  //===================================================================================================================//
  //-- Content rendering (private helpers) --//
  //===================================================================================================================//

  void TerminalUI::renderEpochContent()
  {
    // Pre-estimate whether a scrollbar will be needed to compute the table width.
    // The panel's contentWidth() uses the *current* lines which may be stale, so we
    // estimate from the record/message counts — same approach as the old rebuildEpochLines().
    int contentH = this->epochsPanel.getH() - 2;
    int structuralLines = this->epochRecords.empty() ? 5 : 4;
    int estimatedTotal = static_cast<int>(this->epochRecords.size())
                       + static_cast<int>(this->epochMessages.size())
                       + structuralLines;
    int panelPad = (estimatedTotal > contentH) ? 5 : 4;
    int tableWidth = std::max(40, this->epochsPanel.getW() - panelPad);

    this->epochsTable.setMaxWidth(tableWidth);

    // Choose datetime format based on the computed "Completed At" column width.
    const auto& colWidths = this->epochsTable.getComputedWidths();
    int dateTimeColW = (colWidths.size() > 4) ? colWidths[4] : 19;

    const char* dateFmt = "%Y-%m-%d %H:%M:%S";

    if (dateTimeColW < 19)
      dateFmt = "%m-%d %H:%M:%S";

    if (dateTimeColW < 14)
      dateFmt = "%m-%d %H:%M";

    if (dateTimeColW < 11)
      dateFmt = "%H:%M:%S";

    if (dateTimeColW < 8)
      dateFmt = "%H:%M";

    // Rebuild all rows from the structured epoch records.
    this->epochsTable.clearRows();

    for (const auto& rec : this->epochRecords) {
      char epochStr[16];
      char lossStr[32];
      char valLossStr[32];
      char bestCell[8];
      char dateStr[32];

      snprintf(epochStr, sizeof(epochStr), "%d", rec.epoch);
      snprintf(lossStr, sizeof(lossStr), "%.6f", static_cast<double>(rec.loss));

      if (rec.hasValLoss) {
        snprintf(valLossStr, sizeof(valLossStr), "%.6f", static_cast<double>(rec.valLoss));
      } else {
        snprintf(valLossStr, sizeof(valLossStr), "-");
      }

      snprintf(bestCell, sizeof(bestCell), "%s", rec.isBest ? "X" : "");

      dateStr[0] = '\0';
      std::tm tm_buf{};
      std::tm* tm_info = localtime_r(&rec.completionTime, &tm_buf);

      if (tm_info)
        std::strftime(dateStr, sizeof(dateStr), dateFmt, tm_info);

      this->epochsTable.addRow({epochStr, lossStr, valLossStr, bestCell, dateStr});
    }

    // Render the table to formatted lines, then append stored monitor/status messages.
    auto lines = this->epochsTable.render();

    for (const auto& msg : this->epochMessages)
      lines.push_back(msg);

    this->epochsPanel.setLines(lines);
  }

  //===================================================================================================================//

  void TerminalUI::renderConfigContent()
  {
    if (this->configSections.empty())
      return;

    ulong maxWidth = this->configPanel.getW() > 4 ? static_cast<ulong>(this->configContentWidth()) : 0;
    auto formattedLines = SummaryTable::collectSections(this->configSections, maxWidth);

    // Trim leading empty strings so that offset 0 corresponds to the first non-empty line
    // of content, keeping scroll logic and scrollbar naturally consistent.
    auto firstNonEmpty = std::find_if(formattedLines.begin(), formattedLines.end(),
                                      [](const std::string& s) { return !s.empty(); });
    std::vector<std::string> trimmed(firstNonEmpty, formattedLines.end());

    this->configPanel.setLines(trimmed);
    this->configPanel.scrollState().offset = 0;
  }

  //===================================================================================================================//
  //-- Public API --//
  //===================================================================================================================//

  int TerminalUI::configContentWidth() const
  {
    // Acquire the mutex because this method reads shared state from outside the
    // class — callers like the Runners are not always under a TerminalUI lock when
    // they invoke this.
    std::lock_guard<std::recursive_mutex> lock(this->mutex);

    int cH = this->configPanel.getH() - 2;
    const auto& lines = this->configPanel.getLines();
    int total = static_cast<int>(lines.size());

    // Conservative default: when no lines have been pushed yet, assume the
    // formatted config table will need a scrollbar so the initial render
    // reserves enough space.
    int panelPad = (lines.empty() || total > cH) ? 5 : 4;

    return std::max(1, this->configPanel.getW() - panelPad);
  }

  //===================================================================================================================//

  void TerminalUI::setConfigLines(const std::vector<std::string>& lines)
  {
    std::lock_guard<std::recursive_mutex> lock(this->mutex);

    // Trim leading empty strings so that offset 0 corresponds to the first non-empty line
    // of content, keeping scroll logic and scrollbar naturally consistent.
    auto firstNonEmpty = std::find_if(lines.begin(), lines.end(),
                                      [](const std::string& s) { return !s.empty(); });
    std::vector<std::string> trimmed(firstNonEmpty, lines.end());
    this->configPanel.setLines(trimmed);
    this->configPanel.scrollState().offset = 0;
  }

  //===================================================================================================================//

  void TerminalUI::setConfigSections(const std::vector<SummaryTable::Section>& sections)
  {
    std::lock_guard<std::recursive_mutex> lock(this->mutex);
    this->configSections = sections;
    this->configLinesDirty = true;
  }

  //===================================================================================================================//

  void TerminalUI::setTimingLines(const std::vector<std::string>& lines)
  {
    std::lock_guard<std::recursive_mutex> lock(this->mutex);
    this->timingPanel.setLines(lines);
  }

  //===================================================================================================================//

  void TerminalUI::addEpochLine(const std::string& line)
  {
    std::lock_guard<std::recursive_mutex> lock(this->mutex);
    this->epochMessages.push_back(line);
    this->epochLinesDirty = true;
  }

  //===================================================================================================================//

  void TerminalUI::pushEpochRecord(int epoch, float loss, bool hasValLoss, float valLoss, bool isBest,
                                    std::time_t completionTime)
  {
    std::lock_guard<std::recursive_mutex> lock(this->mutex);

    std::time_t stamp = (completionTime == 0) ? std::time(nullptr) : completionTime;
    this->epochRecords.push_back({epoch, loss, hasValLoss, valLoss, isBest, stamp});

    this->epochLinesDirty = true;
  }

  //===================================================================================================================//

  void TerminalUI::refreshConfigPanel()
  {
    std::lock_guard<std::recursive_mutex> lock(this->mutex);
    this->present(false, false);
  }

  //===================================================================================================================//

  void TerminalUI::requestResize()
  {
    this->resizeRequested.store(1, std::memory_order_relaxed);
  }

  //===================================================================================================================//

  bool TerminalUI::handleResize()
  {
    if (!this->resizeRequested.exchange(0, std::memory_order_relaxed))
      return false;

    this->layout();
    this->epochLinesDirty = true;
    this->configLinesDirty = true;

    // layout() erased and recreated the sub-windows (loadingWin, progressWin).
    // Repaint any overlay (e.g. the loading bar) so it survives the resize even
    // when handleResize() is called from the training callback, which subsequently
    // calls refresh() -> present(false, true) without runOverlay.
    if (this->overlayCallback)
      this->overlayCallback();

    return true;
  }

  //===================================================================================================================//

  void TerminalUI::redraw()
  {
    std::lock_guard<std::recursive_mutex> lock(this->mutex);
    this->resizeRequested.store(0, std::memory_order_relaxed);
    this->layout();
    this->epochLinesDirty = true;
    this->configLinesDirty = true;

    // Full resize redraw: recreate the panels and repaint the loading-bar overlay on top so the
    // loading line doesn't vanish until the next mini-batch tick.
    this->present(true, false);
  }

  //===================================================================================================================//

  void TerminalUI::refresh()
  {
    if (!this->initialized)
      return;

    if (this->resizeRequested.exchange(0, std::memory_order_relaxed)) {
      this->redraw();
      return;
    }

    this->present(false, true);

    int ch = getch();

    if (ch != ERR && ch != KEY_RESIZE && this->handleScrollInput(ch))
      this->present(false, true);
  }

  //===================================================================================================================//
  //-- Input --//
  //===================================================================================================================//

  void TerminalUI::pollInput()
  {
    if (!this->initialized)
      return;

    if (this->resizeRequested.exchange(0, std::memory_order_relaxed)) {
      std::lock_guard<std::recursive_mutex> lock(this->mutex);
      this->redraw();
      return;
    }

    int ch = getch();

    if (ch == ERR)
      return;

    std::lock_guard<std::recursive_mutex> lock(this->mutex);

    if (this->handleScrollInput(ch))
      this->present(false, false);
  }

  //===================================================================================================================//

  bool TerminalUI::handleScrollInput(int ch)
  {
    if (ch == '\t') {
      this->activePanel = (this->activePanel + 1) % 3;
      return true;
    }

    switch (this->activePanel) {
    case 0:
      return this->configPanel.applyScrollInput(ch);
    case 1:
      return this->epochsPanel.applyScrollInput(ch);
    case 2:
      return this->timingPanel.applyScrollInput(ch);
    }

    return false;
  }

} // namespace NN_CLI
