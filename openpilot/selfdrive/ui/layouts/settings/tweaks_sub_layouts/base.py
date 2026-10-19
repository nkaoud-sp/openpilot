import pyray as rl
from collections.abc import Callable

from openpilot.common.params import Params
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller
from openpilot.system.ui.lib.multilang import tr


class TweaksSubLayout(Widget):
  """A tweak's settings sub-page: a Back button over a scroller of controls."""
  def __init__(self, back_callback: Callable):
    super().__init__()
    self._params = Params()
    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_callback)
    self._scroller = Scroller(self._build_items(), line_separator=True, spacing=0)

  def _build_items(self) -> list:
    raise NotImplementedError

  def _render(self, rect):
    self._back_button.set_position(self._rect.x, self._rect.y + 20)
    self._back_button.render()

    content_rect = rl.Rectangle(rect.x, rect.y + self._back_button.rect.height + 40,
                                rect.width, rect.height - self._back_button.rect.height - 40)
    self._scroller.render(content_rect)

  def show_event(self):
    super().show_event()
    self._scroller.show_event()
