#include "NN-CLI_TerminalUI_Window.hpp"

// Qt headers must be included before <curses.h>: curses defines a `timeout`
// macro that breaks Qt declarations parsed after it.
#include <QThread>

#include <algorithm>
#include <clocale>
#include <csignal>
#include <cstdio>
#include <curses.h>

namespace NN_CLI
{

  namespace
  {
    TerminalUI_Window* g_activeWindow = nullptr;

    void sigwinchHandler(int)
    {
      if (g_activeWindow)
        g_activeWindow->requestResize();
    }
  } // namespace

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

    ::initscr();

    ::cbreak();
    ::noecho();
    ::curs_set(0);
    ::keypad(stdscr, TRUE);
    ::nodelay(stdscr, TRUE);

    // Mouse reporting (mousemask) is deliberately NOT enabled: when an SGR
    // mouse sequence arrives truncated (e.g. tty buffer overflow during fast
    // wheel scrolling), ncurses getch() blocks on stdin indefinitely waiting
    // for the rest of it, ignoring nodelay — freezing whichever thread polls
    // input.  Instead, request the terminal's "alternate scroll" mode
    // (DECSET 1007): with mouse reporting off, wheel events on the alternate
    // screen arrive as arrow-key presses, which the panels already handle
    // and which ncurses parses with a bounded ESCDELAY wait.
    ::fputs("\x1b[?1007h", stdout);
    ::fflush(stdout);

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

    this->resize(this->cols, this->rows, 0, 0);

    g_activeWindow = this;
    std::signal(SIGWINCH, sigwinchHandler);

    this->initialized = true;
    return true;
  }

  //===================================================================================================================//

  void TerminalUI_Window::shutdown()
  {
    if (!this->initialized)
      return;

    this->stopUiThread();

    this->initialized = false;
    g_activeWindow = nullptr;
    std::signal(SIGWINCH, SIG_DFL);

    this->children.clear();

    ::endwin();

    // Restore the terminal's default wheel behavior (alternate scroll off).
    ::fputs("\x1b[?1007l", stdout);
    ::fflush(stdout);
  }

  //===================================================================================================================//

  void TerminalUI_Window::startUiThread()
  {
    if (!this->initialized || this->uiThreadRunning.load())
      return;

    this->uiThreadRunning.store(true);
    this->uiThread.reset(QThread::create(&TerminalUI_Window::uiThreadLoop, this));
    this->uiThread->start();
  }

  //===================================================================================================================//

  void TerminalUI_Window::stopUiThread()
  {
    this->uiThreadRunning.store(false);

    if (this->uiThread) {
      this->uiThread->wait();
      this->uiThread.reset();
    }
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
  //-- Resize --//
  //===================================================================================================================//

  void TerminalUI_Window::requestResize()
  {
    this->resizeRequested.store(true, std::memory_order_relaxed);
  }

  //===================================================================================================================//
  //-- Widget overrides --//
  //===================================================================================================================//

  void TerminalUI_Window::draw()
  {
    if (!this->initialized)
      return;

    // Check if a terminal resize was requested by the SIGWINCH handler.
    if (this->resizeRequested.exchange(false, std::memory_order_relaxed)) {
      ::endwin();
      ::refresh();
      ::clear();

      int newRows = ::getmaxy(stdscr);
      int newCols = ::getmaxx(stdscr);

      this->resize(newCols, newRows, 0, 0);
    }

    this->preRender();
    this->render();

    // Drain all pending input events so that buffered keystrokes (including
    // wheel ticks delivered as arrow keys via alternate scroll) are processed
    // in a single pass.  Re-render once at the end if any event was consumed,
    // avoiding per-key repaints.
    bool anyConsumed = false;

    while (this->pollAndDispatchInput())
      anyConsumed = true;

    if (anyConsumed) {
      this->preRender();
      this->render();
    }
  }

  //===================================================================================================================//

  void TerminalUI_Window::render()
  {
    ::touchwin(stdscr);
    ::erase();

    for (auto& child : this->children)
      child->draw();

    ::refresh();
  }

  //===================================================================================================================//

  bool TerminalUI_Window::pollAndDispatchInput()
  {
    int ch = ::getch();

    if (ch == ERR)
      return false;

    return this->handleEvent(ch);
  }

  //===================================================================================================================//

  void TerminalUI_Window::uiThreadLoop()
  {
    // ~30 FPS: frequent enough for smooth progress bars, cheap enough that
    // the periodic redraw is negligible next to the training work.  ncurses
    // diffs the virtual screen internally, so an unchanged frame sends
    // almost nothing to the terminal.
    constexpr unsigned long kFrameIntervalMs = 33;

    while (this->uiThreadRunning.load()) {
      {
        QMutexLocker<QRecursiveMutex> lock(&this->uiMutex);
        this->draw();
      }

      QThread::msleep(kFrameIntervalMs);
    }
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
