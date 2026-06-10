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
    : y(y),
      x(x),
      h(h),
      w(w),
      title(std::move(title)),
      colorPair(colorPair)
  {
  }

  //===================================================================================================================//
  //-- Layout --//
  //===================================================================================================================//

  void TerminalUI_Panel::resize(int y, int x, int h, int w)
  {
    this->y = y;
    this->x = x;
    this->h = h;
    this->w = w;
  }

  //===================================================================================================================//

  void TerminalUI_Panel::setTitle(const std::string& title)
  {
    this->title = title;
  }

  //===================================================================================================================//

  void TerminalUI_Panel::setColorPair(int colorPair)
  {
    this->colorPair = colorPair;
  }

  //===================================================================================================================//
  //-- Content --//
  //===================================================================================================================//

  void TerminalUI_Panel::setLines(const std::vector<std::string>& lines)
  {
    this->lines = lines;
  }

  //===================================================================================================================//

  void TerminalUI_Panel::setAutoScroll(bool autoScroll)
  {
    this->scroll.autoScroll = autoScroll;
  }

  //===================================================================================================================//
  //-- Drawing --//
  //===================================================================================================================//

  void TerminalUI_Panel::drawFrame() const
  {
    if (this->h < 2 || this->w < 2)
      return;

    int titleLen = static_cast<int>(this->title.size());
    int endX = this->x + this->w - 1;

    //-- Top border with optional title --//

    mvaddch(this->y, this->x, ACS_ULCORNER);

    if (titleLen > 0 && 5 + titleLen + 2 < this->w) {
      mvhline(this->y, this->x + 1, ACS_HLINE, 3);
      mvaddstr(this->y, this->x + 4, " ");
      attron(COLOR_PAIR(this->colorPair) | A_BOLD);
      mvaddstr(this->y, this->x + 5, this->title.c_str());
      attroff(COLOR_PAIR(this->colorPair) | A_BOLD);
      int after = 5 + titleLen;
      mvaddstr(this->y, this->x + after, " ");
      int remainingHline = endX - (this->x + after + 1);

      if (remainingHline > 0)
        mvhline(this->y, this->x + after + 1, ACS_HLINE, remainingHline);
    } else {
      mvhline(this->y, this->x + 1, ACS_HLINE, this->w - 2);
    }

    mvaddch(this->y, endX, ACS_URCORNER);

    //-- Side borders (interior blanked) --//

    for (int r = 1; r < this->h - 1; r++) {
      mvaddch(this->y + r, this->x, ACS_VLINE);
      mvhline(this->y + r, this->x + 1, ' ', this->w - 2);
      mvaddch(this->y + r, endX, ACS_VLINE);
    }

    //-- Bottom border --//

    mvaddch(this->y + this->h - 1, this->x, ACS_LLCORNER);
    mvhline(this->y + this->h - 1, this->x + 1, ACS_HLINE, this->w - 2);
    mvaddch(this->y + this->h - 1, endX, ACS_LRCORNER);
  }

  //===================================================================================================================//

  void TerminalUI_Panel::drawContent() const
  {
    int cH = this->contentHeight();

    if (cH <= 0)
      return;

    int total = static_cast<int>(this->lines.size());
    int maxW = this->contentWidth();
    int start = this->scrollOffset();

    for (int i = 0; i < cH; i++) {
      int lineIdx = start + i;

      if (lineIdx >= 0 && lineIdx < total) {
        const std::string& line = this->lines[lineIdx];
        int printLen = std::min(static_cast<int>(line.size()), maxW);

        if (printLen > 0)
          mvaddnstr(this->y + 1 + i, this->x + 2, line.c_str(), printLen);
      }
    }
  }

  //===================================================================================================================//

  void TerminalUI_Panel::drawScrollbar() const
  {
    int cH = this->contentHeight();

    if (cH <= 0)
      return;

    int total = static_cast<int>(this->lines.size());

    if (total <= cH)
      return;

    int col = this->x + this->w - 2;
    int yTop = this->y + 1;
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
    int total = static_cast<int>(this->lines.size());
    int maxScroll = std::max(0, total - cH);

    // Sync offset to the current auto-scroll position before clearing autoScroll,
    // so that the first manual scroll starts from the visible bottom rather than
    // jumping to an unrelated position.
    if (this->scroll.autoScroll)
      this->scroll.offset = maxScroll;

    switch (ch) {
    case KEY_UP:
    case 'k':
      this->scroll.offset = std::max(0, this->scroll.offset - 1);
      break;
    case KEY_DOWN:
    case 'j':
      this->scroll.offset = std::min(maxScroll, this->scroll.offset + 1);
      break;
    case KEY_PPAGE:
      this->scroll.offset = std::max(0, this->scroll.offset - cH);
      break;
    case KEY_NPAGE:
      this->scroll.offset = std::min(maxScroll, this->scroll.offset + cH);
      break;
    case KEY_HOME:
      this->scroll.offset = 0;
      break;
    case KEY_END:
      this->scroll.offset = maxScroll;
      break;
    default:
      return false;
    }

    this->scroll.autoScroll = false;
    return true;
  }

  //===================================================================================================================//
  //-- Accessors --//
  //===================================================================================================================//

  int TerminalUI_Panel::contentWidth() const
  {
    int cH = this->contentHeight();
    int total = static_cast<int>(this->lines.size());
    int pad = (total > cH) ? 5 : 4;

    return std::max(1, this->w - pad);
  }

  //===================================================================================================================//

  int TerminalUI_Panel::contentHeight() const
  {
    return std::max(0, this->h - 2);
  }

  //===================================================================================================================//

  int TerminalUI_Panel::scrollOffset() const
  {
    int cH = this->contentHeight();
    int total = static_cast<int>(this->lines.size());
    int maxScroll = std::max(0, total - cH);

    if (this->scroll.autoScroll)
      return maxScroll;

    return std::clamp(this->scroll.offset, 0, maxScroll);
  }

} // namespace NN_CLI
