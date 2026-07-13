"""Regression tests for the widget's tab/scroll layout."""
from qtpy.QtWidgets import QScrollArea
from napari_convpaint.convpaint_widget import ConvpaintWidget


def test_widget_opens_on_home_tab_and_tabs_scroll(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    widget = ConvpaintWidget(viewer)
    if hasattr(widget, 'ensure_init'):
        widget.ensure_init()
    widget.show()
    qtbot.waitExposed(widget)
    # A fresh widget must open on the first (Home) tab — the scroll-wrap
    # remove/insert dance moves the current index around during construction.
    assert widget.tabs.currentIndex() == 0
    # Every tab is wrapped in a scroll area so content can't be cut off.
    for i in range(widget.tabs.count()):
        assert isinstance(widget.tabs.widget(i), QScrollArea), widget.tabs.tab_names[i]
