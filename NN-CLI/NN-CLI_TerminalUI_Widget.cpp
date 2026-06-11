#include "NN-CLI_TerminalUI_Widget.hpp"

namespace NN_CLI
{

  //===================================================================================================================//
  //-- Core interface --//
  //===================================================================================================================//

  void TerminalUI_Widget::resize(int width, int height, int x, int y)
  {
    this->width = width;
    this->height = height;
    this->x = x;
    this->y = y;
  }

  //===================================================================================================================//

  bool TerminalUI_Widget::handleEvent(int /*ch*/)
  {
    return false;
  }

} // namespace NN_CLI
