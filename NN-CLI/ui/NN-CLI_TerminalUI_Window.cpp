#include "NN-CLI_TerminalUI_Window.hpp"

// Qt headers must be included before <curses.h>: curses defines a `timeout`
// macro that breaks Qt declarations parsed after it.
#include <QTimer>

#include <algorithm>
#include <clocale>
#include <csignal>
#include <cstdio>
#include <unistd.h>
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
    // Guard: when stdin is not a terminal (pipes, tests, CI), skip ncurses
    // entirely so the caller's isInitialized() check gates waitForDismiss()
    // and the console-only fallback path is taken.
    if (!::isatty(STDIN_FILENO)) {
      this->initialized = false;
      return false;
    }

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

    this->stopUiTimer();

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
  //-- Lifecycle --//
  //===================================================================================================================//

  void TerminalUI_Window::startUiTimer()
  {
    if (!this->initialized || this->uiTimer)
      return;

    this->uiTimer = std::unique_ptr<QTimer>(new QTimer());
    this->uiTimer->setInterval(10);

    // Each tick: drain buffered input (non-blocking), repaint only when the
    // widget tree is dirty or a terminal resize is pending.  Same logic as
    // the former uiThreadLoop(), now driven by the main-thread event loop.
    QObject::connect(this->uiTimer.get(), &QTimer::timeout, [this]() {
      QMutexLocker<QRecursiveMutex> lock(&this->uiMutex);

      while (this->pollAndDispatchInput()) {}

      if (this->needsRepaint()) {
        this->draw();
      }
    });

    this->uiTimer->start();
  }

  //===================================================================================================================//

  void TerminalUI_Window::stopUiTimer()
  {
    if (this->uiTimer) {
      this->uiTimer->stop();
      this->uiTimer.reset();
    }
  }

  //===================================================================================================================//
  //-- Layout --//
  //===================================================================================================================//

  void TerminalUI_Window::layoutChildren()
  {
    // Reserve the top row for the title bar and the bottom row(s) for the
    // shortcut bar so children never occupy the rows drawTitleBar() and
    // drawShortcutBar() later paint onto in render().
    int h = std::max(0, this->height - this->shortcutBarHeight() - this->titleBarHeight());

    for (auto& child : this->children)
      child->resize(this->width, h, this->x, this->y + this->titleBarHeight());
  }

  //===================================================================================================================//
  //-- Resize --//
  //===================================================================================================================//

  void TerminalUI_Window::requestResize()
  {
    // Setting an atomic is async-signal-safe.  needsRepaint() checks this flag
    // so the UI thread handles the resize on its next tick.
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
    this->clearDirty();
  }

  //===================================================================================================================//

  void TerminalUI_Window::render()
  {
    ::touchwin(stdscr);
    ::erase();

    for (auto& child : this->children)
      child->draw();

    this->drawTitleBar();
    this->drawShortcutBar();

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

  bool TerminalUI_Window::needsRepaint() const
  {
    return this->isDirtyTree() || this->resizeRequested.load(std::memory_order_relaxed);
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

  //===================================================================================================================//
  //-- Shortcut bar --//
  //===================================================================================================================//

  void TerminalUI_Window::setShortcutBar(const std::string& text)
  {
    this->shortcutBar = text;
    this->markDirty();
  }

  //===================================================================================================================//

  const std::string& TerminalUI_Window::getShortcutBar() const
  {
    return this->shortcutBar;
  }

  //===================================================================================================================//

  int TerminalUI_Window::shortcutBarHeight() const
  {
    return this->shortcutBar.empty() ? 0 : 1;
  }

  //===================================================================================================================//

  void TerminalUI_Window::drawShortcutBar() const
  {
    if (this->shortcutBar.empty())
      return;

    int lastRow = this->rows - 1;

    if (lastRow < 0)
      return;

    ::move(lastRow, 0);
    ::clrtoeol();
    attron(COLOR_PAIR(8));
    int n = std::min<int>(static_cast<int>(this->shortcutBar.size()), this->cols);
    mvaddnstr(lastRow, 0, this->shortcutBar.c_str(), n);
    attroff(COLOR_PAIR(8));
  }

  //===================================================================================================================//
  //-- Title bar --//
  //===================================================================================================================//

  void TerminalUI_Window::setTitleBar(const std::string& text, int colorPair)
  {
    this->titleBar = text;
    this->titleBarColor = colorPair;
    this->markDirty();
  }

  //===================================================================================================================//

  const std::string& TerminalUI_Window::getTitleBar() const
  {
    return this->titleBar;
  }

  //===================================================================================================================//

  int TerminalUI_Window::getTitleBarColor() const
  {
    return this->titleBarColor;
  }

  //===================================================================================================================//

  int TerminalUI_Window::titleBarHeight() const
  {
    return this->titleBar.empty() ? 0 : 1;
  }

  //===================================================================================================================//

  void TerminalUI_Window::drawTitleBar() const
  {
    if (this->titleBar.empty())
      return;

    ::move(0, 0);
    ::clrtoeol();

    int len = static_cast<int>(this->titleBar.size());
    int startX = std::max(0, (this->cols - len) / 2);

    if (this->titleBarColor > 0)
      attron(COLOR_PAIR(this->titleBarColor));

    mvaddnstr(0, startX, this->titleBar.c_str(), std::min(len, this->cols));

    if (this->titleBarColor > 0)
      attroff(COLOR_PAIR(this->titleBarColor));
  }

} // namespace NN_CLI
