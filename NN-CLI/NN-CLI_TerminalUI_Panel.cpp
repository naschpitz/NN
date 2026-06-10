#include "NN-CLI_TerminalUI_Panel.hpp"

#include <algorithm>
#include <curses.h>

namespace NN_CLI
{

  //===================================================================================================================//
  //-- Ctors --//
  //===================================================================================================================//

  TerminalUI_Panel::TerminalUI_Panel() {}

  //===================================================================================================================//

  TerminalUI_Panel::TerminalUI_Panel(int y, int x, int h, int w, std::string title, int colorPair)
    : y_(y),
      x_(x),
      h_(h),
      w_(w),
      title_(std::move(title)),
      colorPair_(colorPair)
  {
  }

  //===================================================================================================================//
  //-- Layout --//
  //===================================================================================================================//

  void TerminalUI_Panel::resize(int y, int x, int h, int w)
  {
    this->y_ = y;
    this->x_ = x;
    this->h_ = h;
    this->w_ = w;
  }

  //===================================================================================================================//

  void TerminalUI_Panel::setTitle(const std::string& title)
  {
    this->title_ = title;
  }

  //===================================================================================================================//

  void TerminalUI_Panel::setColorPair(int colorPair)
  {
    this->colorPair_ = colorPair;
  }

  //===================================================================================================================//
  //-- Content --//
  //===================================================================================================================//

  void TerminalUI_Panel::setLines(const std::vector<std::string>& lines)
  {
    this->lines_ = lines;
  }

  //===================================================================================================================//

  void TerminalUI_Panel::setAutoScroll(bool autoScroll)
  {
    this->scroll_.autoScroll = autoScroll;
  }

  //===================================================================================================================//
  //-- Drawing --//
  //===================================================================================================================//

  void TerminalUI_Panel::drawFrame() const
  {
    if (this->h_ < 2 || this->w_ < 2)
      return;

    int titleLen = static_cast<int>(this->title_.size());
    int endX = this->x_ + this->w_ - 1;

    //-- Top border with optional title --//

    mvaddch(this->y_, this->x_, ACS_ULCORNER);

    if (titleLen > 0 && 5 + titleLen + 2 < this->w_) {
      mvhline(this->y_, this->x_ + 1, ACS_HLINE, 3);
      mvaddstr(this->y_, this->x_ + 4, " ");
      attron(COLOR_PAIR(this->colorPair_) | A_BOLD);
      mvaddstr(this->y_, this->x_ + 5, this->title_.c_str());
      attroff(COLOR_PAIR(this->colorPair_) | A_BOLD);
      int after = 5 + titleLen;
      mvaddstr(this->y_, this->x_ + after, " ");
      int remainingHline = endX - (this->x_ + after + 1);

      if (remainingHline > 0)
        mvhline(this->y_, this->x_ + after + 1, ACS_HLINE, remainingHline);
    } else {
      mvhline(this->y_, this->x_ + 1, ACS_HLINE, this->w_ - 2);
    }

    mvaddch(this->y_, endX, ACS_URCORNER);

    //-- Side borders (interior blanked) --//

    for (int r = 1; r < this->h_ - 1; r++) {
      mvaddch(this->y_ + r, this->x_, ACS_VLINE);
      mvhline(this->y_ + r, this->x_ + 1, ' ', this->w_ - 2);
      mvaddch(this->y_ + r, endX, ACS_VLINE);
    }

    //-- Bottom border --//

    mvaddch(this->y_ + this->h_ - 1, this->x_, ACS_LLCORNER);
    mvhline(this->y_ + this->h_ - 1, this->x_ + 1, ACS_HLINE, this->w_ - 2);
    mvaddch(this->y_ + this->h_ - 1, endX, ACS_LRCORNER);
  }

  //===================================================================================================================//

  void TerminalUI_Panel::drawContent() const
  {
    int cH = this->contentHeight();

    if (cH <= 0)
      return;

    int total = static_cast<int>(this->lines_.size());
    int maxW = this->contentWidth();
    int start = this->scrollOffset();

    for (int i = 0; i < cH; i++) {
      int lineIdx = start + i;

      if (lineIdx >= 0 && lineIdx < total) {
        const std::string& line = this->lines_[lineIdx];
        int printLen = std::min(static_cast<int>(line.size()), maxW);

        if (printLen > 0)
          mvaddnstr(this->y_ + 1 + i, this->x_ + 2, line.c_str(), printLen);
      }
    }
  }

  //===================================================================================================================//

  void TerminalUI_Panel::drawScrollbar() const
  {
    int cH = this->contentHeight();

    if (cH <= 0)
      return;

    int total = static_cast<int>(this->lines_.size());

    if (total <= cH)
      return;

    int col = this->x_ + this->w_ - 2;
    int yTop = this->y_ + 1;
    int scroll = this->scrollOffset();
    int thumb = scroll * (cH - 1) / std::max(1, total - cH);

    for (int i = 0; i < cH; i++)
      mvaddch(yTop + i, col, (i == thumb) ? ACS_CKBOARD : ACS_VLINE);
  }

  //===================================================================================================================//
  //-- Input --//
  //===================================================================================================================//

  bool TerminalUI_Panel::applyScrollInput(int ch)
  {
    int cH = this->contentHeight();
    int total = static_cast<int>(this->lines_.size());
    int maxScroll = std::max(0, total - cH);

    // Sync offset to the current auto-scroll position before clearing autoScroll,
    // so that the first manual scroll starts from the visible bottom rather than
    // jumping to an unrelated position.
    if (this->scroll_.autoScroll)
      this->scroll_.offset = maxScroll;

    switch (ch) {
    case KEY_UP:
    case 'k':
      this->scroll_.offset = std::max(0, this->scroll_.offset - 1);
      break;
    case KEY_DOWN:
    case 'j':
      this->scroll_.offset = std::min(maxScroll, this->scroll_.offset + 1);
      break;
    case KEY_PPAGE:
      this->scroll_.offset = std::max(0, this->scroll_.offset - cH);
      break;
    case KEY_NPAGE:
      this->scroll_.offset = std::min(maxScroll, this->scroll_.offset + cH);
      break;
    case KEY_HOME:
      this->scroll_.offset = 0;
      break;
    case KEY_END:
      this->scroll_.offset = maxScroll;
      break;
    default:
      return false;
    }

    this->scroll_.autoScroll = false;
    return true;
  }

  //===================================================================================================================//
  //-- Accessors --//
  //===================================================================================================================//

  int TerminalUI_Panel::contentWidth() const
  {
    int cH = this->contentHeight();
    int total = static_cast<int>(this->lines_.size());
    int pad = (total > cH) ? 5 : 4;

    return std::max(1, this->w_ - pad);
  }

  //===================================================================================================================//

  int TerminalUI_Panel::contentHeight() const
  {
    return std::max(0, this->h_ - 2);
  }

  //===================================================================================================================//

  int TerminalUI_Panel::scrollOffset() const
  {
    int cH = this->contentHeight();
    int total = static_cast<int>(this->lines_.size());
    int maxScroll = std::max(0, total - cH);

    if (this->scroll_.autoScroll)
      return maxScroll;

    return std::clamp(this->scroll_.offset, 0, maxScroll);
  }

} // namespace NN_CLI
