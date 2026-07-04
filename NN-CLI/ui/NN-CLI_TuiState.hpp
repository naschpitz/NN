#ifndef NN_CLI_TUISTATE_HPP
#define NN_CLI_TUISTATE_HPP

//===================================================================================================================//
// Process-global flag tracking whether the terminal UI (ncurses) owns the
// terminal.  Set by TerminalUI_Window::init() once ncurses is up, cleared by
// shutdown().
//
// Any code path that would write to stdout/cerr consults isTuiActive() and
// bails: raw bytes from a worker thread interleave with ncurses' escape
// sequences and corrupt the screen layout.  When the TUI is NOT active (no
// TTY — pipes, tests, CI) the flag stays false and all console output flows
// normally, preserving the headless contract.
//===================================================================================================================//

namespace NN_CLI
{
  bool isTuiActive() noexcept;
  void setTuiActive(bool active) noexcept;
}

#endif
