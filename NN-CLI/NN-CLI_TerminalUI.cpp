#include "NN-CLI_TerminalUI.hpp"

#include <algorithm>
#include <clocale>
#include <csignal>
#include <cstring>
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

    this->resized_ = true;

    int minTimingW = 20;
    int minLeftW = 40;

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

    this->configH_ = std::max(3, std::min(remaining - 5, remaining * 35 / 100));
    this->epochsH_ = std::max(3, remaining - this->configH_);

    this->configY_ = 0;
    this->trainingY_ = this->configH_;
    this->epochsY_ = this->trainingY_ + this->trainingH_;

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

    int loadW = std::max(1, this->leftWidth_ - 2);

    this->loadingWin_ = newwin(1, loadW, this->trainingY_ + 1, 1);
    this->progressWin_ = newwin(2, loadW, this->trainingY_ + 2, 1);
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

  void TerminalUI::drawAllPanels()
  {
    touchwin(stdscr);
    erase();

    int contentH = 0;
    int maxW = this->leftWidth_ - 4;

    int cfgColor = (this->activePanel_ == 0) ? 3 : 2;
    int epColor = (this->activePanel_ == 1) ? 3 : 2;
    int timColor = (this->activePanel_ == 2) ? 3 : 2;

    //-- Left column: Config panel --//
    this->drawPanelFrame(this->configY_, this->configH_, 0, this->leftWidth_, "Config", cfgColor);
    contentH = this->configH_ - 2;

    for (int i = 0; i < contentH; i++) {
      int lineIdx = this->configScroll_ + i;

      if (lineIdx >= 0 && lineIdx < static_cast<int>(this->configLines_.size())) {
        const std::string& line = this->configLines_[lineIdx];
        int printLen = std::min(static_cast<int>(line.size()), maxW);

        if (printLen > 0)
          mvaddnstr(this->configY_ + 1 + i, 2, line.c_str(), printLen);
      }
    }

    if (static_cast<int>(this->configLines_.size()) > contentH) {
      int indicatorCol = this->leftWidth_ - 2;
      int totalCfg = static_cast<int>(this->configLines_.size());
      int scrollFrac = this->configScroll_ * (contentH - 1) / std::max(1, totalCfg - contentH);

      for (int i = 0; i < contentH; i++) {
        if (i == scrollFrac)
          mvaddch(this->configY_ + 1 + i, indicatorCol, ACS_CKBOARD);
        else
          mvaddch(this->configY_ + 1 + i, indicatorCol, ACS_VLINE);
      }
    }

    //-- Left column: Training panel --//
    this->drawPanelFrame(this->trainingY_, this->trainingH_, 0, this->leftWidth_, "Training", 2);

    //-- Left column: Epochs panel --//
    this->drawPanelFrame(this->epochsY_, this->epochsH_, 0, this->leftWidth_, "Epochs", epColor);
    contentH = this->epochsH_ - 2;
    int totalLines = static_cast<int>(this->epochLines_.size());
    int maxScroll = std::max(0, totalLines - contentH);
    int start = this->epochScroll_;

    if (this->epochsAutoScroll_)
      start = maxScroll;

    if (start < 0)
      start = 0;

    if (start > maxScroll)
      start = maxScroll;

    for (int i = 0; i < contentH && start + i < totalLines; i++) {
      const std::string& line = this->epochLines_[start + i];
      int printLen = std::min(static_cast<int>(line.size()), maxW);

      if (printLen > 0)
        mvaddnstr(this->epochsY_ + 1 + i, 2, line.c_str(), printLen);
    }

    if (totalLines > contentH) {
      int indicatorCol = this->leftWidth_ - 2;
      int scrollFrac = start * (contentH - 1) / std::max(1, maxScroll);

      for (int i = 0; i < contentH; i++) {
        if (i == scrollFrac)
          mvaddch(this->epochsY_ + 1 + i, indicatorCol, ACS_CKBOARD);
        else
          mvaddch(this->epochsY_ + 1 + i, indicatorCol, ACS_VLINE);
      }
    }

    //-- Right column: Timing panel (full-height) --//
    if (this->timingWin_) {
      int timingH = this->helpY_;
      int timingX = this->leftWidth_;

      this->drawPanelFrame(0, timingH, timingX, this->timingWidth_, "Timing", timColor);
      contentH = timingH - 2;
      int timingMaxW = this->timingWidth_ - 4;
      int timingTotal = static_cast<int>(this->timingLines_.size());
      int timingMaxScroll = std::max(0, timingTotal - contentH);
      int timingStart = this->timingScroll_;

      if (timingStart < 0)
        timingStart = 0;

      if (timingStart > timingMaxScroll)
        timingStart = timingMaxScroll;

      for (int i = 0; i < contentH && timingStart + i < timingTotal; i++) {
        const std::string& line = this->timingLines_[timingStart + i];
        int printLen = std::min(static_cast<int>(line.size()), timingMaxW);

        if (printLen > 0)
          mvaddnstr(1 + i, timingX + 2, line.c_str(), printLen);
      }

      if (timingTotal > contentH) {
        int indicatorCol = this->cols_ - 2;
        int scrollFrac = timingStart * (contentH - 1) / std::max(1, timingMaxScroll);

        for (int i = 0; i < contentH; i++) {
          if (i == scrollFrac)
            mvaddch(1 + i, indicatorCol, ACS_CKBOARD);
          else
            mvaddch(1 + i, indicatorCol, ACS_VLINE);
        }
      }
    }

    //-- Help bar --//
    mvaddch(this->helpY_, 0, ACS_LLCORNER);
    mvhline(this->helpY_, 1, ACS_HLINE, this->cols_ - 2);
    mvaddch(this->helpY_, this->cols_ - 1, ACS_LRCORNER);

    attron(COLOR_PAIR(2) | A_BOLD);
    mvaddstr(this->helpY_, 3, "Tab: select panel  jk/arrows: scroll  PgUp/PgDn: page  Home/End: jump");
    attroff(COLOR_PAIR(2) | A_BOLD);
  }

  //===================================================================================================================//
  //-- Public API --//
  //===================================================================================================================//

  void TerminalUI::setConfigLines(const std::vector<std::string>& lines)
  {
    std::lock_guard<std::recursive_mutex> lock(this->mutex_);
    this->configLines_ = lines;
    this->configScroll_ = 0;
  }

  void TerminalUI::setTimingLines(const std::vector<std::string>& lines)
  {
    std::lock_guard<std::recursive_mutex> lock(this->mutex_);
    this->timingLines_ = lines;
  }

  void TerminalUI::addEpochLine(const std::string& line)
  {
    std::lock_guard<std::recursive_mutex> lock(this->mutex_);
    this->epochLines_.push_back(line);

    if (this->epochsAutoScroll_)
      this->epochScroll_ = std::max(0, static_cast<int>(this->epochLines_.size()) - (this->epochsH_ - 2));

    this->drawAllPanels();
    wnoutrefresh(stdscr);

    if (this->loadingWin_)
      wnoutrefresh(this->loadingWin_);

    if (this->progressWin_)
      wnoutrefresh(this->progressWin_);

    doupdate();
  }

  void TerminalUI::refreshConfigPanel()
  {
    std::lock_guard<std::recursive_mutex> lock(this->mutex_);
    this->drawAllPanels();
    wnoutrefresh(stdscr);

    if (this->loadingWin_)
      wnoutrefresh(this->loadingWin_);

    if (this->progressWin_)
      wnoutrefresh(this->progressWin_);

    doupdate();
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
    this->drawAllPanels();
    wnoutrefresh(stdscr);
    return true;
  }

  void TerminalUI::redraw()
  {
    std::lock_guard<std::recursive_mutex> lock(this->mutex_);
    this->resizeRequested_.store(0, std::memory_order_relaxed);
    this->layout();
    this->drawAllPanels();
    wnoutrefresh(stdscr);

    if (this->loadingWin_)
      wnoutrefresh(this->loadingWin_);

    if (this->progressWin_)
      wnoutrefresh(this->progressWin_);

    doupdate();
  }

  void TerminalUI::refresh()
  {
    if (!this->initialized_)
      return;

    if (this->resizeRequested_.exchange(0, std::memory_order_relaxed)) {
      this->redraw();
      return;
    }

    this->drawAllPanels();
    wnoutrefresh(stdscr);

    if (this->loadingWin_) {
      touchwin(this->loadingWin_);
      wnoutrefresh(this->loadingWin_);
    }

    if (this->progressWin_) {
      touchwin(this->progressWin_);
      wnoutrefresh(this->progressWin_);
    }

    doupdate();

    int ch = getch();

    if (ch != ERR && ch != KEY_RESIZE && this->handleScrollInput(ch)) {
      this->drawAllPanels();
      wnoutrefresh(stdscr);

      if (this->loadingWin_) {
        touchwin(this->loadingWin_);
        wnoutrefresh(this->loadingWin_);
      }

      if (this->progressWin_) {
        touchwin(this->progressWin_);
        wnoutrefresh(this->progressWin_);
      }

      doupdate();
    }
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

    if (this->handleScrollInput(ch)) {
      this->drawAllPanels();
      wnoutrefresh(stdscr);

      if (this->loadingWin_)
        wnoutrefresh(this->loadingWin_);

      if (this->progressWin_)
        wnoutrefresh(this->progressWin_);

      doupdate();
    }
  }

  bool TerminalUI::handleScrollInput(int ch)
  {
    switch (ch) {
    case '\t':
      this->activePanel_ = (this->activePanel_ + 1) % 3;
      return true;
    }

    switch (this->activePanel_) {
    case 0: { // Config
      int contentH = this->configH_ - 2;
      int maxScroll = std::max(0, static_cast<int>(this->configLines_.size()) - contentH);

      switch (ch) {
      case KEY_UP:
      case 'k':
        this->configScroll_ = std::max(0, this->configScroll_ - 1);
        return true;
      case KEY_DOWN:
      case 'j':
        this->configScroll_ = std::min(maxScroll, this->configScroll_ + 1);
        return true;
      case KEY_PPAGE:
        this->configScroll_ = std::max(0, this->configScroll_ - contentH);
        return true;
      case KEY_NPAGE:
        this->configScroll_ = std::min(maxScroll, this->configScroll_ + contentH);
        return true;
      case KEY_HOME:
        this->configScroll_ = 0;
        return true;
      case KEY_END:
        this->configScroll_ = maxScroll;
        return true;
      }

      break;
    }

    case 1: { // Epochs
      int contentH = this->epochsH_ - 2;
      int totalLines = static_cast<int>(this->epochLines_.size());
      int maxScroll = std::max(0, totalLines - contentH);

      switch (ch) {
      case KEY_UP:
      case 'k':
        this->epochScroll_ = std::max(0, this->epochScroll_ - 1);
        this->epochsAutoScroll_ = false;
        return true;
      case KEY_DOWN:
      case 'j':
        this->epochScroll_ = std::min(maxScroll, this->epochScroll_ + 1);
        this->epochsAutoScroll_ = false;
        return true;
      case KEY_PPAGE:
        this->epochScroll_ = std::max(0, this->epochScroll_ - contentH);
        this->epochsAutoScroll_ = false;
        return true;
      case KEY_NPAGE:
        this->epochScroll_ = std::min(maxScroll, this->epochScroll_ + contentH);
        this->epochsAutoScroll_ = false;
        return true;
      case KEY_HOME:
        this->epochScroll_ = 0;
        this->epochsAutoScroll_ = false;
        return true;
      case KEY_END:
        this->epochScroll_ = maxScroll;
        this->epochsAutoScroll_ = false;
        return true;
      }

      break;
    }

    case 2: { // Timing
      int contentH = this->helpY_ - 2;
      int totalLines = static_cast<int>(this->timingLines_.size());
      int maxScroll = std::max(0, totalLines - contentH);

      switch (ch) {
      case KEY_UP:
      case 'k':
        this->timingScroll_ = std::max(0, this->timingScroll_ - 1);
        return true;
      case KEY_DOWN:
      case 'j':
        this->timingScroll_ = std::min(maxScroll, this->timingScroll_ + 1);
        return true;
      case KEY_PPAGE:
        this->timingScroll_ = std::max(0, this->timingScroll_ - contentH);
        return true;
      case KEY_NPAGE:
        this->timingScroll_ = std::min(maxScroll, this->timingScroll_ + contentH);
        return true;
      case KEY_HOME:
        this->timingScroll_ = 0;
        return true;
      case KEY_END:
        this->timingScroll_ = maxScroll;
        return true;
      }

      break;
    }
    }

    return false;
  }

} // namespace NN_CLI
