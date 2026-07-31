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


def test_output_layers_do_not_steal_selection(make_napari_viewer):
    """Creating segmentation/probabilities/features layers must leave the
    active layer untouched — napari selects new layers by default, which
    would steer the user's brush into the output layer instead of the
    annotations layer they were painting on."""
    import numpy as np
    from napari_convpaint.convpaint_widget import ConvpaintWidget

    viewer = make_napari_viewer()
    w = ConvpaintWidget(viewer)
    w.ensure_init()
    viewer.add_image(np.random.random((64, 64)), name='img')
    w._on_add_annot_layer()
    annot = viewer.layers['annotations']
    viewer.layers.selection.active = annot

    w._check_create_segmentation_layer()
    assert viewer.layers.selection.active is annot
    w._check_create_probas_layer(2)
    assert viewer.layers.selection.active is annot
    w._check_create_features_layer(4)
    assert viewer.layers.selection.active is annot
    assert {'segmentation'} <= {l.name for l in viewer.layers}


def test_classes_tab_value_model(make_napari_viewer):
    """Value-based class rows: placeholder icons before any layer; 'Remove
    class' targets the SELECTED class (grayed without a selection or at the
    two-class floor) and never renumbers others; sync adds exactly the painted
    values (sparse — no gap filling)."""
    import numpy as np
    from napari_convpaint.convpaint_widget import ConvpaintWidget

    viewer = make_napari_viewer()
    w = ConvpaintWidget(viewer)
    w.ensure_init()

    # Startup: NO classes — two empty placeholder slots, Remove grayed
    assert w.class_rows == []
    assert len(w._placeholder_rows) == 2
    assert all(not r['icon'].pixmap().isNull() for r in w._placeholder_rows)
    assert not w.remove_class_btn.isEnabled()
    # Create the two classes the rest of this test works with
    w._on_add_class(text='Background', value=1)
    w._on_add_class(text='Foreground', value=2)

    viewer.add_image(np.random.random((64, 64)), name='img')
    w._on_add_annot_layer()
    annot = viewer.layers['annotations']

    # A selected real class is removable even at two classes (no floor)
    annot.selected_label = 1
    assert w.remove_class_btn.isEnabled()

    # Sparse sync: painting 7 adds EXACTLY value 7 (no gap rows 3..6)
    annot.data[5:10, 5:10] = 7
    w._on_sync_classes_from_annotations()
    assert [r['value'] for r in w.class_rows] == [1, 2, 7]

    # Remove the selected middle-by-value class: value 2 goes, 7 keeps its value
    annot.selected_label = 2
    assert w.remove_class_btn.isEnabled()
    w._on_remove_class()
    assert [r['value'] for r in w.class_rows] == [1, 7]
    # ...and its annotations were erased, others kept
    assert not (annot.data == 2).any()
    assert (annot.data == 7).any()

    # Selection now points at a value with no row -> Remove grayed
    assert annot.selected_label == 2
    assert not w.remove_class_btn.isEnabled()

    # Add class assigns max+1
    w._on_add_class()
    assert [r['value'] for r in w.class_rows] == [1, 7, 8]


def test_classes_tab_no_floor_and_placeholders(make_napari_viewer):
    """No class-count floor; placeholders are pure UI (not clickable classes)
    that pad the display to two slots; typing into a placeholder's name field
    creates the class in place."""
    import numpy as np
    from napari_convpaint.convpaint_widget import ConvpaintWidget

    viewer = make_napari_viewer()
    w = ConvpaintWidget(viewer)
    w.ensure_init()
    viewer.add_image(np.random.random((64, 64)), name='img')
    w._on_add_annot_layer()
    annot = viewer.layers['annotations']
    w._on_add_class(text='Background', value=1)
    w._on_add_class(text='Foreground', value=2)

    # Remove both classes, one by one — no floor
    annot.selected_label = 2
    w._on_remove_class()
    assert [r['value'] for r in w.class_rows] == [1]
    assert len(w._placeholder_rows) == 1
    annot.selected_label = 1
    w._on_remove_class()
    assert w.class_rows == []
    assert len(w._placeholder_rows) == 2

    # Placeholders: editable italic 'add class' fields, no class behavior
    for ph in w._placeholder_rows:
        assert ph['name'].isEnabled()
        assert ph['name'].placeholderText() == 'add class'
        assert 'italic' in ph['name'].styleSheet()
    # Selecting a placeholder's value (e.g. via the label spinbox) grays Remove
    annot.selected_label = w._placeholder_rows[0]['value']
    assert not w.remove_class_btn.isEnabled()

    # Typing into a placeholder creates the class with that text, in place
    ph = w._placeholder_rows[0]
    v = ph['value']
    w._on_placeholder_name_edited(ph, 'Nu')
    assert [r['value'] for r in w.class_rows] == [v]
    assert w._row_for_value(v)['name'].text() == 'Nu'


def test_classes_selectable_without_annotations_layer(make_napari_viewer):
    """Deleting the annotations layer must not strand the class list: the
    widget keeps its own selection memory, so swatch clicks still select and
    Remove still works with no layer present."""
    import numpy as np
    from napari_convpaint.convpaint_widget import ConvpaintWidget

    viewer = make_napari_viewer()
    w = ConvpaintWidget(viewer)
    w.ensure_init()
    viewer.add_image(np.random.random((64, 64)), name='img')
    w._on_add_annot_layer()
    annot = viewer.layers['annotations']
    w._on_add_class(text='Background', value=1)
    w._on_add_class(text='Foreground', value=2)
    annot.selected_label = 2                     # mirrored into widget memory
    viewer.layers.remove('annotations')          # layer gone

    assert w._selected_class_value() == 2        # selection survives
    w._update_selected_class_highlight()
    assert w.remove_class_btn.isEnabled()
    w._on_remove_class()                         # removable without a layer
    assert [r['value'] for r in w.class_rows] == [1]

    # Swatch click selects widget-side with no layer at all
    w._on_class_swatch_clicked(1)
    assert w._selected_class_value() == 1
    assert w.remove_class_btn.isEnabled()


def test_selected_class_highlight_follows_selected_label(make_napari_viewer):
    """The Classes tab outlines the row matching the annotations layer's
    selected label (by VALUE), in both directions."""
    import numpy as np
    from napari_convpaint.convpaint_widget import ConvpaintWidget

    viewer = make_napari_viewer()
    w = ConvpaintWidget(viewer)
    w.ensure_init()
    viewer.add_image(np.random.random((64, 64)), name='img')
    w._on_add_annot_layer()
    annot = viewer.layers['annotations']
    w._on_add_class(value=1)
    w._on_add_class(value=2)

    def outlined():
        return [r['value'] for r in w.class_rows
                if 'transparent' not in r['icon'].styleSheet()]

    annot.selected_label = 2
    assert outlined() == [2]
    annot.selected_label = 1
    assert outlined() == [1]


def test_class_value_limit(make_napari_viewer, tmp_path):
    """Class values are capped at 255 (annotation/segmentation data are
    uint8): adding beyond the limit is refused, the Add button grays out at
    the limit, and a CSV with too-high values fails without side effects."""
    import pytest
    from napari_convpaint.convpaint_widget import ConvpaintWidget

    viewer = make_napari_viewer()
    w = ConvpaintWidget(viewer)
    w.ensure_init()

    # Adding a value above the limit is a no-op
    w._on_add_class(value=300)
    assert w._class_values() == []

    # At the limit the Add button grays out; below it stays enabled
    w._on_add_class(value=254)
    assert w.add_class_btn.isEnabled()
    w._on_add_class(value=255)
    assert w._class_values() == [254, 255]
    assert not w.add_class_btn.isEnabled()

    # A CSV holding a too-high value raises and leaves the classes untouched
    bad = tmp_path / 'bad.csv'
    bad.write_text('index,name\n1,ok\n300,too high\n')
    with pytest.raises(ValueError, match='255'):
        w.import_class_names_csv(str(bad))
    assert w._class_values() == [254, 255]

    # A valid sparse CSV still round-trips
    good = tmp_path / 'good.csv'
    good.write_text('index,name\n1,bg\n7,rare\n')
    w.import_class_names_csv(str(good))
    assert w._class_values() == [1, 7]
    assert w.add_class_btn.isEnabled()
