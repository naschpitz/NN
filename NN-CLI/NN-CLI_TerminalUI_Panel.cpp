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

  TerminalUI_Panel::TerminalUI_Panel(const std::string& title, int colorPair) : title(title), colorPair(colorPair) {}

  //===================================================================================================================//

  TerminalUI_Panel::TerminalUI_Panel(int y, int x, int h, int w, std::string title, int colorPair)
    : title(std::move(title)),
      colorPair(colorPair)
  {
    // Map (y, x, h, w) to Widget's (x, y, width, height).
    this->x = x;
    this->y = y;
    this->width = w;
    this->height = h;
  }

  //===================================================================================================================//
  //-- Layout --//
  //===================================================================================================================//

  void TerminalUI_Panel::resize(int width, int height, int x, int y)
  {
    TerminalUI_Widget::resize(width, height, x, y);
    this->layoutChildren();
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

  void TerminalUI_Panel::draw()
  {
    this->drawFrame();
    this->drawContent();
    this->drawScrollbar();

    for (auto& child : this->children)
      child->draw();
  }

  //===================================================================================================================//

  void TerminalUI_Panel::drawFrame() const
  {
    if (this->height < 2 || this->width < 2)
      return;

    int titleLen = static_cast<int>(this->title.size());
    int endX = this->x + this->width - 1;

    //-- Top border with optional title --//

    mvaddch(this->y, this->x, ACS_ULCORNER);

    if (titleLen > 0 && 5 + titleLen + 2 < this->width) {
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
      mvhline(this->y, this->x + 1, ACS_HLINE, this->width - 2);
    }

    mvaddch(this->y, endX, ACS_URCORNER);

    //-- Side borders (interior blanked) --//

    for (int r = 1; r < this->height - 1; r++) {
      mvaddch(this->y + r, this->x, ACS_VLINE);
      mvhline(this->y + r, this->x + 1, ' ', this->width - 2);
      mvaddch(this->y + r, endX, ACS_VLINE);
    }

    //-- Bottom border --//

    mvaddch(this->y + this->height - 1, this->x, ACS_LLCORNER);
    mvhline(this->y + this->height - 1, this->x + 1, ACS_HLINE, this->width - 2);
    mvaddch(this->y + this->height - 1, endX, ACS_LRCORNER);
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

    int col = this->x + this->width - 2;
    int yTop = this->y + 1;
    int scr = this->scrollOffset();
    int thumb = scr * (cH - 1) / std::max(1, total - cH);

    for (int i = 0; i < cH; i++)
      mvaddch(yTop + i, col, (i == thumb) ? ACS_CKBOARD : ACS_VLINE);
  }

  //===================================================================================================================//
  //-- Input --//
  //===================================================================================================================//

  bool TerminalUI_Panel::handleEvent(int ch)
  {
    if (this->applyScrollInput(ch))
      return true;

    for (auto& child : this->children) {
      if (child->handleEvent(ch))
        return true;
    }

    return false;
  }

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
  //-- Child management --//
  //===================================================================================================================//

  void TerminalUI_Panel::addChild(std::unique_ptr<TerminalUI_Widget> child)
  {
    if (child)
      this->children.push_back(std::move(child));
  }

  //===================================================================================================================//

  std::unique_ptr<TerminalUI_Widget> TerminalUI_Panel::removeChild(int index)
  {
    if (index < 0 || index >= static_cast<int>(this->children.size()))
      return nullptr;

    auto removed = std::move(this->children[index]);
    this->children.erase(this->children.begin() + index);

    return removed;
  }

  //===================================================================================================================//

  TerminalUI_Widget* TerminalUI_Panel::getChild(int index) const
  {
    if (index < 0 || index >= static_cast<int>(this->children.size()))
      return nullptr;

    return this->children[index].get();
  }

  //===================================================================================================================//
  //-- Layout (children) --//
  //===================================================================================================================//

  void TerminalUI_Panel::layoutChildren()
  {
    int cx = this->x + 2;
    int cy = this->y + 1;
    int cw = std::max(0, this->width - 4);
    int ch = std::max(0, this->height - 2);

    for (auto& child : this->children)
      child->resize(cw, ch, cx, cy);
  }

  //===================================================================================================================//
  //-- Accessors --//
  //===================================================================================================================//

  int TerminalUI_Panel::contentWidth() const
  {
    int cH = this->contentHeight();
    int total = static_cast<int>(this->lines.size());
    int pad = (total > cH) ? 5 : 4;

    return std::max(1, this->width - pad);
  }

  //===================================================================================================================//

  int TerminalUI_Panel::contentHeight() const
  {
    return std::max(0, this->height - 2);
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
