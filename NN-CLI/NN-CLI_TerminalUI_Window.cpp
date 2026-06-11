#include "NN-CLI_TerminalUI_Window.hpp"

#include <algorithm>
#include <clocale>
#include <curses.h>

namespace NN_CLI
{

  //===================================================================================================================//
  //-- Ctors / Dtors --//
  //===================================================================================================================//

  TerminalUI_Window::TerminalUI_Window() {}

  //===================================================================================================================//

  TerminalUI_Window::~TerminalUI_Window()
  {
    this->shutdown();
  }

  //===================================================================================================================//
  //-- Lifecycle --//
  //===================================================================================================================//

  bool TerminalUI_Window::init()
  {
    if (this->initialized)
      return true;

    ::setlocale(LC_ALL, "");

    FILE* tty = ::fopen("/dev/tty", "r+");

    if (!tty)
      return false;

    SCREEN* screen = ::newterm(nullptr, tty, tty);

    if (!screen) {
      ::fclose(tty);
      return false;
    }

    ::set_term(screen);

    ::cbreak();
    ::noecho();
    ::curs_set(0);
    ::keypad(stdscr, TRUE);
    ::nodelay(stdscr, TRUE);

    if (::has_colors()) {
      ::start_color();
      ::use_default_colors();

      ::init_pair(1, COLOR_GREEN, -1);
      ::init_pair(2, COLOR_CYAN, -1);
      ::init_pair(3, COLOR_YELLOW, -1);
      ::init_pair(4, COLOR_RED, -1);
      ::init_pair(5, COLOR_WHITE, -1);
      ::init_pair(6, COLOR_GREEN, -1);
      ::init_pair(7, COLOR_MAGENTA, -1);
      ::init_pair(8, COLOR_BLUE, -1);
    }

    this->rows = ::getmaxy(stdscr);
    this->cols = ::getmaxx(stdscr);

    TerminalUI_Widget::resize(this->cols, this->rows, 0, 0);

    this->initialized = true;
    return true;
  }

  //===================================================================================================================//

  void TerminalUI_Window::shutdown()
  {
    if (!this->initialized)
      return;

    this->initialized = false;
    this->children.clear();

    ::endwin();
  }

  //===================================================================================================================//
  //-- Child management --//
  //===================================================================================================================//

  void TerminalUI_Window::addChild(std::unique_ptr<TerminalUI_Widget> child)
  {
    if (child)
      this->children.push_back(std::move(child));
  }

  //===================================================================================================================//

  std::unique_ptr<TerminalUI_Widget> TerminalUI_Window::removeChild(int index)
  {
    if (index < 0 || index >= static_cast<int>(this->children.size()))
      return nullptr;

    auto removed = std::move(this->children[index]);
    this->children.erase(this->children.begin() + index);

    return removed;
  }

  //===================================================================================================================//

  TerminalUI_Widget* TerminalUI_Window::getChild(int index) const
  {
    if (index < 0 || index >= static_cast<int>(this->children.size()))
      return nullptr;

    return this->children[index].get();
  }

  //===================================================================================================================//
  //-- Layout --//
  //===================================================================================================================//

  void TerminalUI_Window::layoutChildren()
  {
    for (auto& child : this->children)
      child->resize(this->width, this->height, this->x, this->y);
  }

  //===================================================================================================================//
  //-- Widget overrides --//
  //===================================================================================================================//

  void TerminalUI_Window::draw()
  {
    if (!this->initialized)
      return;

    ::touchwin(stdscr);
    ::erase();

    for (auto& child : this->children)
      child->draw();

    ::refresh();
  }

  //===================================================================================================================//

  void TerminalUI_Window::resize(int width, int height, int x, int y)
  {
    TerminalUI_Widget::resize(width, height, x, y);

    this->rows = height;
    this->cols = width;

    this->layoutChildren();
  }

  //===================================================================================================================//

  bool TerminalUI_Window::handleEvent(int ch)
  {
    for (auto& child : this->children) {
      if (child->handleEvent(ch))
        return true;
    }

    return false;
  }

} // namespace NN_CLI
