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

  TerminalUI::TerminalUI() {}

  TerminalUI::~TerminalUI()
  {
    this->shutdown();
  }

  //===================================================================================================================//

  bool TerminalUI::init()
  {
    if (this->initialized_)
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

    this->initialized_ = true;
    return true;
  }

  //===================================================================================================================//

  void TerminalUI::shutdown()
  {
    if (!this->initialized_)
      return;

    this->initialized_ = false;
    g_activeUI = nullptr;
    std::signal(SIGWINCH, SIG_DFL);
    std::signal(SIGINT, SIG_DFL);
    std::signal(SIGTERM, SIG_DFL);

    if (this->loadingWin_) {
      delwin(this->loadingWin_);
      this->loadingWin_ = nullptr;
    }

    if (this->progressWin_) {
      delwin(this->progressWin_);
      this->progressWin_ = nullptr;
    }

    if (this->timingWin_) {
      delwin(this->timingWin_);
      this->timingWin_ = nullptr;
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

    this->rows_ = getmaxy(stdscr);
    this->cols_ = getmaxx(stdscr);

    int minTimingW = 20;
    int minLeftW = 68;

    if (this->cols_ < minLeftW + minTimingW) {
      this->leftWidth_ = this->cols_;
      this->timingWidth_ = 0;
    } else {
      int idealTimingW = this->cols_ * 35 / 100;
      this->timingWidth_ = std::max(minTimingW, std::min(this->cols_ - minLeftW, idealTimingW));
      this->leftWidth_ = this->cols_ - this->timingWidth_;
    }

    int screenRows = this->rows_ - 1;
    this->helpY_ = screenRows;

    this->trainingH_ = 5;
    int remaining = std::max(8, screenRows - this->trainingH_);

    this->configH_ = std::max(3, std::min(remaining - 5, remaining * 55 / 100));
    this->epochsH_ = std::max(3, remaining - this->configH_);

    this->trainingY_ = 0;
    this->epochsY_ = this->trainingY_ + this->trainingH_;
    this->configY_ = this->epochsY_ + this->epochsH_;

    if (this->progressWin_) {
      delwin(this->progressWin_);
      this->progressWin_ = nullptr;
    }

    if (this->loadingWin_) {
      delwin(this->loadingWin_);
      this->loadingWin_ = nullptr;
    }

    if (this->timingWin_) {
      delwin(this->timingWin_);
      this->timingWin_ = nullptr;
    }

    int loadW = std::max(1, this->leftWidth_ - 4);

    this->loadingWin_ = newwin(1, loadW, this->trainingY_ + 1, 2);
    this->progressWin_ = newwin(2, loadW, this->trainingY_ + 2, 2);
    this->timingWin_ = (this->timingWidth_ > 0) ? newwin(screenRows, this->timingWidth_, 0, this->leftWidth_) : nullptr;

    if (this->loadingWin_) {
      werase(this->loadingWin_);
      touchwin(this->loadingWin_);
    }

    if (this->progressWin_) {
      werase(this->progressWin_);
      touchwin(this->progressWin_);
    }

    if (this->timingWin_) {
      werase(this->timingWin_);
      touchwin(this->timingWin_);
    }

    if (this->resizeCallback_)
      this->resizeCallback_();
  }

  //===================================================================================================================//
  //-- Drawing primitives --//
  //===================================================================================================================//

  void TerminalUI::drawPanelFrame(int y, int h, const char* title, int titleColor)
  {
    this->drawPanelFrame(y, h, 0, this->cols_, title, titleColor);
  }

  void TerminalUI::drawPanelFrame(int y, int h, int x, int w, const char* title, int titleColor)
  {
    if (y < 0 || y + h > this->rows_ || h < 2)
      return;

    int titleLen = static_cast<int>(std::strlen(title));
    int endX = x + w - 1;

    mvaddch(y, x, ACS_ULCORNER);

    if (titleLen > 0 && 5 + titleLen + 2 < w) {
      mvhline(y, x + 1, ACS_HLINE, 3);
      mvaddstr(y, x + 4, " ");
      attron(COLOR_PAIR(titleColor) | A_BOLD);
      mvaddstr(y, x + 5, title);
      attroff(COLOR_PAIR(titleColor) | A_BOLD);
      int after = 5 + titleLen;
      mvaddstr(y, x + after, " ");
      mvhline(y, x + after + 1, ACS_HLINE, endX - (x + after + 1));
    } else {
      mvhline(y, x + 1, ACS_HLINE, w - 2);
    }

    mvaddch(y, endX, ACS_URCORNER);

    for (int r = 1; r < h - 1; r++) {
      mvaddch(y + r, x, ACS_VLINE);
      mvhline(y + r, x + 1, ' ', w - 2);
      mvaddch(y + r, endX, ACS_VLINE);
    }

    mvaddch(y + h - 1, x, ACS_LLCORNER);
    mvhline(y + h - 1, x + 1, ACS_HLINE, w - 2);
    mvaddch(y + h - 1, endX, ACS_LRCORNER);
  }

  //===================================================================================================================//
  //-- Full redraw --//
  //===================================================================================================================//

  int TerminalUI::configContentWidth() const
  {
    // Acquire the mutex because this method reads shared state (configLines_,
    // configH_, leftWidth_) from outside the class — callers like the Runners
    // are not always under a TerminalUI lock when they invoke this.
    std::lock_guard<std::recursive_mutex> lock(this->mutex_);

    int panelPad = 4;

    if (this->configLines_.empty()) {
      // Conservative default: when no lines have been pushed yet, assume the
      // formatted config table will need a scrollbar so the initial render
      // reserves enough space.  Otherwise the "no-scrollbar" width (pad=4) is
      // tried first, the table overflows, and the renderer clips because
      // scrollbar space was not pre-reserved.
      panelPad = 5;
    } else if (static_cast<int>(this->configLines_.size()) > this->configH_ - 2) {
      panelPad = 5;
    }

    return std::max(1, this->leftWidth_ - panelPad);
  }

  void TerminalUI::drawAllPanels()
  {
    touchwin(stdscr);
    erase();

    int contentH = 0;

    int cfgColor = (this->activePanel_ == 0) ? 3 : 2;
    int epColor = (this->activePanel_ == 1) ? 3 : 2;
    int timColor = (this->activePanel_ == 2) ? 3 : 2;

    //-- Left column: Config panel --//
    this->drawPanelFrame(this->configY_, this->configH_, 0, this->leftWidth_, "Configuration", cfgColor);
    contentH = this->configH_ - 2;

    int configTotal = static_cast<int>(this->configLines_.size());
    int configMaxW = (configTotal <= contentH) ? (this->leftWidth_ - 4) : (this->leftWidth_ - 5);

    for (int i = 0; i < contentH; i++) {
      int lineIdx = this->config_.offset + i;

      if (lineIdx >= 0 && lineIdx < configTotal) {
        const std::string& line = this->configLines_[lineIdx];
        int printLen = std::min(static_cast<int>(line.size()), configMaxW);

        if (printLen > 0)
          mvaddnstr(this->configY_ + 1 + i, 2, line.c_str(), printLen);
      }
    }

    this->drawScrollbar(this->leftWidth_ - 2, this->configY_ + 1, contentH, this->config_.offset,
                        static_cast<int>(this->configLines_.size()));

    //-- Left column: Training panel --//
    this->drawPanelFrame(this->trainingY_, this->trainingH_, 0, this->leftWidth_, "Training", 2);

    //-- Left column: Epochs panel --//
    this->drawPanelFrame(this->epochsY_, this->epochsH_, 0, this->leftWidth_, "Epochs", epColor);
    contentH = this->epochsH_ - 2;
    int totalLines = static_cast<int>(this->epochLines_.size());
    int epochsMaxW = (totalLines <= contentH) ? (this->leftWidth_ - 4) : (this->leftWidth_ - 5);
    int maxScroll = std::max(0, totalLines - contentH);
    int start = this->epochs_.autoScroll ? maxScroll : std::clamp(this->epochs_.offset, 0, maxScroll);

    for (int i = 0; i < contentH && start + i < totalLines; i++) {
      const std::string& line = this->epochLines_[start + i];
      int printLen = std::min(static_cast<int>(line.size()), epochsMaxW);

      if (printLen > 0)
        mvaddnstr(this->epochsY_ + 1 + i, 2, line.c_str(), printLen);
    }

    this->drawScrollbar(this->leftWidth_ - 2, this->epochsY_ + 1, contentH, start, totalLines);

    //-- Right column: Timing panel (full-height) --//
    if (this->timingWin_) {
      int timingH = this->helpY_;
      int timingX = this->leftWidth_;

      this->drawPanelFrame(0, timingH, timingX, this->timingWidth_, "Timing", timColor);
      contentH = timingH - 2;
      int timingTotal = static_cast<int>(this->timingLines_.size());
      // The scrollbar (when needed) sits at cols_-2 inside the right-padding space —
      // content always uses timingWidth_-4 so the profiler can format at a stable width.
      int timingMaxW = this->timingWidth_ - 4;
      int timingMaxScroll = std::max(0, timingTotal - contentH);
      int timingStart = std::clamp(this->timing_.offset, 0, timingMaxScroll);

      for (int i = 0; i < contentH && timingStart + i < timingTotal; i++) {
        const std::string& line = this->timingLines_[timingStart + i];
        int printLen = std::min(static_cast<int>(line.size()), timingMaxW);

        if (printLen > 0)
          mvaddnstr(1 + i, timingX + 2, line.c_str(), printLen);
      }

      this->drawScrollbar(this->cols_ - 2, 1, contentH, timingStart, timingTotal);
    }

    //-- Help bar --//
    mvaddch(this->helpY_, 0, ACS_LLCORNER);
    mvhline(this->helpY_, 1, ACS_HLINE, this->cols_ - 2);
    mvaddch(this->helpY_, this->cols_ - 1, ACS_LRCORNER);

    attron(COLOR_PAIR(2) | A_BOLD);
    mvaddstr(this->helpY_, 3, "Tab: select panel  jk/arrows: scroll  PgUp/PgDn: page  Home/End: jump");
    attroff(COLOR_PAIR(2) | A_BOLD);
  }

  void TerminalUI::drawScrollbar(int col, int yTop, int contentH, int scroll, int total)
  {
    if (total <= contentH)
      return;

    int thumb = scroll * (contentH - 1) / std::max(1, total - contentH);

    for (int i = 0; i < contentH; i++)
      mvaddch(yTop + i, col, (i == thumb) ? ACS_CKBOARD : ACS_VLINE);
  }

  //===================================================================================================================//
  //-- Compositing --//
  //===================================================================================================================//

  void TerminalUI::present(bool runOverlay, bool touchSub)
  {
    if (this->epochLinesDirty_) {
      this->rebuildEpochLines();
      this->epochLinesDirty_ = false;
    }
    this->drawAllPanels();
    wnoutrefresh(stdscr);

    // The loading bar (and any overlay) lives in its own window; repaint it on top of the panels
    // after layout() recreated/erased the windows.
    if (runOverlay && this->overlayCallback_)
      this->overlayCallback_();

    // drawAllPanels() blanks the stdscr cells beneath the sub-windows, so when a sub-window's own
    // content is unchanged we must touch it to force a full re-copy on top.
    if (this->loadingWin_) {
      if (touchSub)
        touchwin(this->loadingWin_);

      wnoutrefresh(this->loadingWin_);
    }

    if (this->progressWin_) {
      if (touchSub)
        touchwin(this->progressWin_);

      wnoutrefresh(this->progressWin_);
    }

    doupdate();
  }

  //===================================================================================================================//
  //-- Public API --//
  //===================================================================================================================//

  void TerminalUI::setConfigLines(const std::vector<std::string>& lines)
  {
    std::lock_guard<std::recursive_mutex> lock(this->mutex_);

    // Trim leading empty strings so that offset 0 corresponds to the first non-empty line
    // of content, keeping scroll logic and scrollbar naturally consistent.
    auto firstNonEmpty = std::find_if(lines.begin(), lines.end(),
                                      [](const std::string& s) { return !s.empty(); });
    this->configLines_.assign(firstNonEmpty, lines.end());
    this->config_.offset = 0;
  }

  void TerminalUI::setTimingLines(const std::vector<std::string>& lines)
  {
    std::lock_guard<std::recursive_mutex> lock(this->mutex_);
    this->timingLines_ = lines;
  }

  void TerminalUI::addEpochLine(const std::string& line)
  {
    std::lock_guard<std::recursive_mutex> lock(this->mutex_);
    this->epochMessages_.push_back(line);
    this->epochLinesDirty_ = true;
  }

  //===================================================================================================================//

  void TerminalUI::pushEpochRecord(int epoch, float loss, bool hasValLoss, float valLoss, bool isBest,
                                    std::time_t completionTime)
  {
    std::lock_guard<std::recursive_mutex> lock(this->mutex_);

    std::time_t stamp = (completionTime == 0) ? std::time(nullptr) : completionTime;
    this->epochRecords_.push_back({epoch, loss, hasValLoss, valLoss, isBest, stamp});

    this->epochLinesDirty_ = true;
  }

  //===================================================================================================================//

  void TerminalUI::rebuildEpochLines()
  {
    // Rebuild the entire table with borders from scratch
    this->epochLines_.clear();

    // Compute dynamic column widths to fill the available panel space
    int contentH = this->epochsH_ - 2;
    int structuralLines = this->epochRecords_.empty() ? 5 : 4;
    int estimatedTotal = static_cast<int>(this->epochRecords_.size())
                       + static_cast<int>(this->epochMessages_.size())
                       + structuralLines;
    int panelPad = (estimatedTotal > contentH) ? 5 : 4;
    int maxW = std::max(40, static_cast<int>(this->leftWidth_ - panelPad));
    // Width accounting
    int overhead         = 16; // non-data chars: "| " + " | " + " | " + " | " + " | " + " |"
    int reservedLossWidth = 16; // minimum space reserved for loss + valLoss columns (8+8)
    int dataW = std::max(0, maxW - overhead);

    int epochW = 5;
    int bestW = 4;
    int remaining = dataW - epochW - bestW;

    // Give datetime at most 19 chars, minimum 4, leaving at least 8+8 for loss columns
    int dateTimeW = std::max(4, std::min(19, remaining - reservedLossWidth));

    // Split the rest equally between loss and val loss
    int lossValSpace = remaining - dateTimeW;
    int lossW = std::max(4, lossValSpace / 2);
    int valLossW = std::max(4, lossValSpace - lossW);

    // Separator line helper
    auto sep = [&]() -> std::string {
      return "+" + std::string(epochW + 2, '-') + "+" + std::string(lossW + 2, '-') + "+" +
             std::string(valLossW + 2, '-') + "+" + std::string(bestW + 2, '-') + "+" +
             std::string(dateTimeW + 2, '-') + "+";
    };

    // Top border
    this->epochLines_.push_back(sep());

    // Row length is the same for header and data rows — compute once.
    const int lineLen = epochW + lossW + valLossW + bestW + dateTimeW + overhead;

    // Header row
    {
      std::vector<char> hdr(lineLen + 1);

      // GCC -Wformat-truncation: GCC cannot prove the buffer is large enough because
      // column widths are runtime-computed. snprintf is always safe (truncates rather
      // than overflows). For typical terminal widths the buffer is exact; on very
      // narrow terminals, header strings may exceed their column width, causing
      // cosmetic truncation — not a safety issue.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
      snprintf(hdr.data(), hdr.size(), "| %-*s | %*s | %*s | %-*s | %-*s |", epochW, "Epoch", lossW, "Loss",
               valLossW, "Val Loss", bestW, "Best", dateTimeW, "Completed At");
#pragma GCC diagnostic pop

      this->epochLines_.push_back(hdr.data());
    }

    // Header/data separator
    this->epochLines_.push_back(sep());

    // Data rows (one empty line if no records yet)
    if (this->epochRecords_.empty()) {
      std::vector<char> row(lineLen + 1);

      // GCC -Wformat-truncation: same rationale as the header snprintf above.
      // Column widths are runtime-computed; snprintf is always safe (truncates
      // rather than overflows). On very narrow terminals cosmetic truncation may
      // occur — not a safety issue.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
      snprintf(row.data(), row.size(), "| %*s | %*s | %*s | %-*s | %-*s |", epochW, "", lossW, "",
               valLossW, "", bestW, "", dateTimeW, "");
#pragma GCC diagnostic pop
      this->epochLines_.push_back(row.data());
    } else {
      for (const auto& rec : this->epochRecords_) {
        char epochStr[16], lossStr[32], valLossStr[32], bestCell[8];
        snprintf(epochStr, sizeof(epochStr), "%d", rec.epoch);
        snprintf(lossStr, sizeof(lossStr), "%.6f", static_cast<double>(rec.loss));

        if (rec.hasValLoss) {
          snprintf(valLossStr, sizeof(valLossStr), "%.6f", static_cast<double>(rec.valLoss));
        } else {
          snprintf(valLossStr, sizeof(valLossStr), "-");
        }

        snprintf(bestCell, sizeof(bestCell), "%s", rec.isBest ? "X" : "");

        char dateStr[32] = "";
        std::tm tm_buf{};
        std::tm* tm_info = localtime_r(&rec.completionTime, &tm_buf);

        if (tm_info) {
          const char* dateFmt = "%Y-%m-%d %H:%M:%S";

          if (dateTimeW < 19)
            dateFmt = "%m-%d %H:%M:%S";

          if (dateTimeW < 14)
            dateFmt = "%m-%d %H:%M";

          if (dateTimeW < 11)
            dateFmt = "%H:%M:%S";

          if (dateTimeW < 8)
            dateFmt = "%H:%M";
          std::strftime(dateStr, sizeof(dateStr), dateFmt, tm_info);
        }

        std::vector<char> row(lineLen + 1);

        // GCC -Wformat-truncation: same rationale as the header snprintf above.
        // Column widths are runtime-computed; snprintf is always safe (truncates
        // rather than overflows). On very narrow terminals cosmetic truncation may
        // occur — not a safety issue.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
        snprintf(row.data(), row.size(), "| %*s | %*s | %*s | %-*s | %-*s |", epochW, epochStr, lossW, lossStr,
                 valLossW, valLossStr, bestW, bestCell, dateTimeW, dateStr);
#pragma GCC diagnostic pop

        this->epochLines_.push_back(row.data());
      }
    }

    // Bottom border
    this->epochLines_.push_back(sep());

    // Append stored monitor/status messages below the table
    for (const auto& msg : this->epochMessages_)
      this->epochLines_.push_back(msg);

    // Auto-scroll offset - header is now part of the data, so use epochsH_ - 2
    if (this->epochs_.autoScroll)
      this->epochs_.offset = std::max(0, static_cast<int>(this->epochLines_.size()) - std::max(0, this->epochsH_ - 2));
  }

  void TerminalUI::refreshConfigPanel()
  {
    std::lock_guard<std::recursive_mutex> lock(this->mutex_);
    this->present(false, false);
  }

  void TerminalUI::requestResize()
  {
    this->resizeRequested_.store(1, std::memory_order_relaxed);
  }

  bool TerminalUI::handleResize()
  {
    if (!this->resizeRequested_.exchange(0, std::memory_order_relaxed))
      return false;

    this->layout();
    this->epochLinesDirty_ = true;
    return true;
  }

  void TerminalUI::redraw()
  {
    std::lock_guard<std::recursive_mutex> lock(this->mutex_);
    this->resizeRequested_.store(0, std::memory_order_relaxed);
    this->layout();
    this->epochLinesDirty_ = true;

    // Full resize redraw: recreate the panels and repaint the loading-bar overlay on top so the
    // loading line doesn't vanish until the next mini-batch tick.
    this->present(true, false);
  }

  void TerminalUI::refresh()
  {
    if (!this->initialized_)
      return;

    if (this->resizeRequested_.exchange(0, std::memory_order_relaxed)) {
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
    if (!this->initialized_)
      return;

    if (this->resizeRequested_.exchange(0, std::memory_order_relaxed)) {
      std::lock_guard<std::recursive_mutex> lock(this->mutex_);
      this->redraw();
      return;
    }

    int ch = getch();

    if (ch == ERR)
      return;

    std::lock_guard<std::recursive_mutex> lock(this->mutex_);

    if (this->handleScrollInput(ch))
      this->present(false, false);
  }

  bool TerminalUI::applyScroll(ScrollState& s, int ch, int contentH, int total)
  {
    int maxScroll = std::max(0, total - contentH);

    switch (ch) {
    case KEY_UP:
    case 'k':
      s.offset = std::max(0, s.offset - 1);
      break;
    case KEY_DOWN:
    case 'j':
      s.offset = std::min(maxScroll, s.offset + 1);
      break;
    case KEY_PPAGE:
      s.offset = std::max(0, s.offset - contentH);
      break;
    case KEY_NPAGE:
      s.offset = std::min(maxScroll, s.offset + contentH);
      break;
    case KEY_HOME:
      s.offset = 0;
      break;
    case KEY_END:
      s.offset = maxScroll;
      break;
    default:
      return false;
    }

    s.autoScroll = false;
    return true;
  }

  bool TerminalUI::handleScrollInput(int ch)
  {
    if (ch == '\t') {
      this->activePanel_ = (this->activePanel_ + 1) % 3;
      return true;
    }

    switch (this->activePanel_) {
    case 0:
      return this->applyScroll(this->config_, ch, this->configH_ - 2, static_cast<int>(this->configLines_.size()));
    case 1:
      return this->applyScroll(this->epochs_, ch, this->epochsH_ - 2, static_cast<int>(this->epochLines_.size()));
    case 2:
      return this->applyScroll(this->timing_, ch, this->helpY_ - 2, static_cast<int>(this->timingLines_.size()));
    }

    return false;
  }

} // namespace NN_CLI
