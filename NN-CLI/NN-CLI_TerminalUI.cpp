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
        g_activeUI->redraw();
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

    endwin();
  }

  //===================================================================================================================//
  //-- Layout --//
  //===================================================================================================================//

  void TerminalUI::layout()
  {
    timeout(0);
    int ch = getch();

    if (ch == KEY_RESIZE)
      resize_term(0, 0);

    this->rows_ = getmaxy(stdscr);
    this->cols_ = getmaxx(stdscr);

    if (this->rows_ < 16 || this->cols_ < 40)
      return;

    int screenRows = this->rows_ - 1;
    this->helpY_ = screenRows;

    this->trainingH_ = 6;
    int remaining = screenRows - this->trainingH_;

    this->configH_ = std::max(5, std::min(remaining - 10, remaining * 35 / 100));
    this->timingH_ = std::max(5, std::min(remaining - this->configH_ - 3, 16));
    this->epochsH_ = remaining - this->configH_ - this->timingH_;

    if (this->epochsH_ < 3) {
      this->epochsH_ = 3;
      this->timingH_ = std::max(5, remaining - this->configH_ - this->epochsH_);
    }

    this->configY_ = 0;
    this->trainingY_ = this->configH_;
    this->timingY_ = this->trainingY_ + this->trainingH_;
    this->epochsY_ = this->timingY_ + this->timingH_;

    if (this->progressWin_) {
      delwin(this->progressWin_);
      this->progressWin_ = nullptr;
    }

    if (this->loadingWin_) {
      delwin(this->loadingWin_);
      this->loadingWin_ = nullptr;
    }

    this->loadingWin_ = newwin(1, this->cols_ - 2, this->trainingY_ + 1, 1);
    this->progressWin_ = newwin(3, this->cols_ - 2, this->trainingY_ + 2, 1);
  }

  //===================================================================================================================//
  //-- Drawing primitives --//
  //===================================================================================================================//

  void TerminalUI::drawPanelFrame(int y, int h, const char* title, int titleColor)
  {
    if (y < 0 || y + h > this->rows_ || h < 2)
      return;

    int w = this->cols_;
    int titleLen = static_cast<int>(std::strlen(title));

    mvaddch(y, 0, ACS_ULCORNER);

    if (titleLen > 0 && 5 + titleLen + 2 < w) {
      mvhline(y, 1, ACS_HLINE, 3);
      mvaddstr(y, 4, " ");
      attron(COLOR_PAIR(titleColor) | A_BOLD);
      mvaddstr(y, 5, title);
      attroff(COLOR_PAIR(titleColor) | A_BOLD);
      int after = 5 + titleLen;
      mvaddstr(y, after, " ");
      mvhline(y, after + 1, ACS_HLINE, w - after - 2);
    } else {
      mvhline(y, 1, ACS_HLINE, w - 2);
    }

    mvaddch(y, w - 1, ACS_URCORNER);

    for (int r = 1; r < h - 1; r++) {
      mvaddch(y + r, 0, ACS_VLINE);
      mvhline(y + r, 1, ' ', w - 2);
      mvaddch(y + r, w - 1, ACS_VLINE);
    }

    mvaddch(y + h - 1, 0, ACS_LLCORNER);
    mvhline(y + h - 1, 1, ACS_HLINE, w - 2);
    mvaddch(y + h - 1, w - 1, ACS_LRCORNER);
  }

  //===================================================================================================================//
  //-- Full redraw --//
  //===================================================================================================================//

  void TerminalUI::drawAllPanels()
  {
    erase();

    int contentH = 0;
    int maxW = this->cols_ - 4;
    int cfgColor = this->epochsActive_ ? 2 : 3;
    int epColor = this->epochsActive_ ? 3 : 2;

    //--- Config panel ---//
    this->drawPanelFrame(this->configY_, this->configH_, "Config", cfgColor);
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

    // Scroll indicator for config
    if (static_cast<int>(this->configLines_.size()) > contentH) {
      int indicatorCol = this->cols_ - 2;
      int scrollFrac =
        this->configScroll_ * (contentH - 1) / std::max(1, static_cast<int>(this->configLines_.size()) - contentH);

      for (int i = 0; i < contentH; i++) {
        if (i == scrollFrac)
          mvaddch(this->configY_ + 1 + i, indicatorCol, ACS_CKBOARD);
        else
          mvaddch(this->configY_ + 1 + i, indicatorCol, ACS_VLINE);
      }
    }

    //--- Training panel (border only; content drawn by ProgressBar into progressWin_) ---//
    this->drawPanelFrame(this->trainingY_, this->trainingH_, "Training", 2);

    //--- Timing panel ---//
    this->drawPanelFrame(this->timingY_, this->timingH_, "Timing", 2);
    contentH = this->timingH_ - 2;

    for (int i = 0; i < contentH && i < static_cast<int>(this->timingLines_.size()); i++) {
      const std::string& line = this->timingLines_[i];
      int printLen = std::min(static_cast<int>(line.size()), maxW);

      if (printLen > 0)
        mvaddnstr(this->timingY_ + 1 + i, 2, line.c_str(), printLen);
    }

    //--- Epochs panel ---//
    this->drawPanelFrame(this->epochsY_, this->epochsH_, "Epochs", epColor);
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

    // Scroll indicator for epochs
    if (totalLines > contentH) {
      int indicatorCol = this->cols_ - 2;
      int scrollFrac = start * (contentH - 1) / std::max(1, maxScroll);

      for (int i = 0; i < contentH; i++) {
        if (i == scrollFrac)
          mvaddch(this->epochsY_ + 1 + i, indicatorCol, ACS_CKBOARD);
        else
          mvaddch(this->epochsY_ + 1 + i, indicatorCol, ACS_VLINE);
      }
    }

    //--- Help bar ---//
    mvaddch(this->helpY_, 0, ACS_LLCORNER);
    mvhline(this->helpY_, 1, ACS_HLINE, this->cols_ - 2);
    mvaddch(this->helpY_, this->cols_ - 1, ACS_LRCORNER);

    attron(COLOR_PAIR(2) | A_BOLD);
    mvaddstr(this->helpY_, 3, "Tab: switch panel  jk/arrows: scroll  PgUp/PgDn: page  Home/End: jump");
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

  void TerminalUI::redraw()
  {
    std::lock_guard<std::recursive_mutex> lock(this->mutex_);
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
      this->epochsActive_ = !this->epochsActive_;
      return true;
    }

    if (this->epochsActive_) {
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
    } else {
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
    }

    return false;
  }

} // namespace NN_CLI
