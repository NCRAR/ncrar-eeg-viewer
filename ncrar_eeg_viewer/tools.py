import pyqtgraph as pg
import numpy as np

from enaml.qt import QtCore, QtWidgets


class CrosshairTool:
    def __init__(self, presenter):
        self.presenter = presenter
        self.vb = self.presenter.viewbox
        self.pc = None
        self.x_data = None
        self.y_data = None
        self.proxy = None
        self.enabled = False

        # Create UI elements
        self.widgets = {
            'vline': pg.InfiniteLine(angle=90, movable=False, pen='k'),
            'hline': pg.InfiniteLine(angle=0, movable=False, pen='k'),
            'dot': pg.ScatterPlotItem(size=10, pen='r', brush='k'),
        }

        for item in self.widgets.values():
            self.vb.addItem(item, ignoreBounds=True)
            item.hide()

    def mouse_clicked(self, click_event):
        if not self.enabled:
            return
        if click_event.button() != QtCore.Qt.LeftButton:
            return
        if self.pc is None:
            return
        text, ok = QtWidgets.QInputDialog.getText(None, 'Add Marker', 'Label for point')
        if ok:
            x, y = self.widgets['dot'].getData()
            self.presenter.add_analysis(self.pc, text, x[0], y[0])
            self.disable()
        click_event.accept()

    def enable(self, pc):
        """Points the crosshair to a plot item and initializes the signal connection."""
        if self.pc is not None:
            self.pc.has_crosshairs = False
        self.enabled = True
        self.pc = pc
        self.pc.has_crosshairs = True
        self.x_data, self.y_data = self.pc.plot_item.getData()

        # Show items now that we have a target
        for item in self.widgets.values():
            item.show()

        # Connect the signal only if it hasn't been connected yet
        if self.proxy is None:
            scene = self.vb.scene()
            scene.sigMouseClicked.connect(self.mouse_clicked)
            self.proxy = pg.SignalProxy(scene.sigMouseMoved, rateLimit=60, slot=self.update)

    def disable(self):
        if self.pc is not None:
            self.pc.has_crosshairs = False
        self.enabled = False
        for item in self.widgets.values():
            item.hide()

    def update(self, event):
        if not self.enabled:
            return
        if self.pc is None:
            return

        # Use sceneBoundingRect to ensure we are inside the specific ViewBox
        if self.vb.sceneBoundingRect().contains(event[0]):
            mousePoint = self.vb.mapSceneToView(event[0])
            x_val = mousePoint.x()

            # Binary search for efficiency
            idx = np.searchsorted(self.x_data, x_val)
            idx = max(0, min(idx, len(self.x_data) - 1))

            snap_x = self.x_data[idx]
            snap_y = self.y_data[idx]

            # Update positions
            self.widgets['vline'].setPos(snap_x)
            self.widgets['hline'].setPos(snap_y)
            self.widgets['dot'].setData([{'pos': (snap_x, snap_y)}])
