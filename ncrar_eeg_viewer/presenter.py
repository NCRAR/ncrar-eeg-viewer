from copy import deepcopy
import importlib
import itertools
import json
from pathlib import Path

from atom.api import observe, Atom, Bool, Int, Float, Dict, List, Value, Str, Typed
from enaml.colors import Color
from matplotlib.figure import Figure
import mne.io
import numpy as np
import pandas as pd
from scipy import signal

import pyqtgraph as pg

mne.set_log_level(verbose=False)
pg.setConfigOption('foreground', 'k')

from biosemi_enaml.electrode_selector import BiosemiElectrodeSelector

from . import io
from .tools import CrosshairTool


def get_color_cycle(name, n):
    module_name, cmap_name = name.rsplit('.', 1)
    module = importlib.import_module(module_name)

    # This generates a LinearSegmetnedColormap instance that interpolates to
    # the requested number of colors. We can then extract these colors by
    # calling the colormap with a mapping of 0 ... 1 where the number of values
    # in the array is the number of colors we need (spaced equally along 0 ...
    # 1).
    cmap = getattr(module, cmap_name).mpl_colormap.resampled(n)
    for i in np.linspace(0, 1, n):
        yield tuple(int(v * 255) for v in cmap(i))


class PlotConfig(Atom):

    presenter = Value()
    channel = Str()
    event = Str()
    event_label = Str()
    visible = Bool()
    color = Value()
    plot_item = Value()
    has_crosshairs = Bool(False)

    def _observe_visible(self, event):
        if self.plot_item is None:
            return
        self.plot_item.setVisible(self.visible)

    def _observe_color(self, event):
        if self.plot_item is None:
            return
        c = (self.color.red, self.color.green, self.color.blue, self.color.alpha)
        pen = pg.mkPen(c, width=3)
        self.plot_item.setPen(pen)


class AnalysisItem(Atom):

    channel = Str()
    event_label = Str()
    label = Str()
    latency = Float()
    amplitude = Float()


class Presenter(Atom):

    selector = Typed(BiosemiElectrodeSelector)
    figure = Typed(Figure)
    axes = Value()
    eeg_plot = Value()

    filename = Value()
    epochs = Value()

    trigger_channel = Str('Erg1').tag(persist=True, step='annotate')
    filt_lb = Float(0.1).tag(persist=True, step='filter')
    filt_ub = Float(300).tag(persist=True, step='filter')
    time_lb = Float(-200e-3).tag(persist=True, step='epoch')
    time_ub = Float(800e-3).tag(persist=True, step='epoch')
    event_config = Dict().tag(persist=True, step='plot_config')
    selected_electrodes = List().tag(persist=True, step='plot_config')
    reference_electrodes = List().tag(persist=True, step='plot')

    container = Value()
    viewbox = Value()
    crosshairs = Value()
    current_state = Dict()
    plot_config = List(Typed(PlotConfig)).tag(step='plot')

    raw = Value().tag(step='annotate')
    raw_annotated = Value().tag(step='filter')
    raw_filtered = Value().tag(step='epoch')
    epochs = Value().tag(step='plot')

    analysis = List()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = BiosemiElectrodeSelector(
            n_channels=64,
            include_exg=False,
            select_mode='multi'
        )
        self.selector.observe('reference', self._update_reference)
        self.selector.observe('selected', self._update_selected)
        self.make_container()

    def _update_reference(self, event):
        with self.suppress_notifications():
            self.reference_electrodes = self.selector.reference.copy()
        self.update_plots()

    def _update_selected(self, event):
        with self.suppress_notifications():
            self.selected_electrodes = self.selector.selected.copy()
        self.update_plots()

    def _observe_reference_electrodes(self, event):
        self.selector.reference = self.reference_electrodes

    def _observe_selected_electrodes(self, event):
        self.selector.selected = self.selected_electrodes

    def set_channels(self, channels):
        if isinstance(channels, str):
            channels = [channels]

    def set_reference_channels(self, channels):
        if isinstance(channels, str):
            channels = {channels}
        self.selector.reference = set(channels)

    def make_container(self):
        container = pg.GraphicsLayout()
        container.setSpacing(10)
        viewbox = pg.ViewBox(enableMenu=True)
        viewbox.setBackgroundColor('white')
        x_axis = pg.AxisItem('bottom')
        x_axis.linkToView(viewbox)
        x_axis.setLabel('Time', units='s')
        y_axis = pg.AxisItem('left')
        y_axis.setLabel('Signal', units='V')
        y_axis.linkToView(viewbox)
        container.addItem(y_axis, 0, 0)
        container.addItem(x_axis, 1, 1)
        container.addItem(viewbox, 0, 1)
        self.container = container
        self.viewbox = viewbox
        self.crosshairs = CrosshairTool(self)

    def save_analysis(self, filename):
        filename = Path(filename).with_suffix('.csv')
        result = []
        for item in self.analysis:
            result.append({
                'channel': item.channel,
                'event_label': item.event_label,
                'label': item.label,
                'latency': item.latency,
                'amplitude': item.amplitude,
            })
        result = pd.DataFrame(result)
        result.to_csv(filename, index=False)

    def load_analysis(self, filename):
        filename = Path(filename)
        items = []
        for _, row in pd.read_csv(filename).iterrows():
            items.append(AnalysisItem(**row))
        self.analysis = items

    def save_config(self, filename):
        filename = Path(filename)
        config = {}
        for name, member in self.members().items():
            if member.metadata and member.metadata.get('persist'):
                config[name] = getattr(self, name)
        filename.with_suffix('.json').write_text(json.dumps(config, indent=4))

    def load_config(self, filename):
        filename = Path(filename)
        config = json.loads(filename.read_text())
        for name, value in config.items():
            setattr(self, name, value)

    def get_settings(self, step):
        settings = {}
        for name, member in self.members().items():
            if member.metadata and member.metadata.get('step') == step:
                if name == 'event_config':
                    settings[name] = deepcopy(getattr(self, name))
                else:
                    settings[name] = getattr(self, name)
        if len(settings) == 0:
            raise ValueError('No settings for step %s', step)
        return settings

    def settings_changed(self, step):
        return self.current_state.get(step) != self.get_settings(step)

    def update_settings(self, step):
        self.current_state[step] = self.get_settings(step)

    def load_file(self, filename):
        self.filename = filename
        self.raw = mne.io.read_raw_bdf(filename, preload=True, verbose=False)
        extra = set(self.raw.info.ch_names) - set(self.selector.coords.index.values)
        self.selector.extra = sorted(extra - {'Status', 'Erg1', 'Erg2'})
        self.find_triggers()
        self.reprocess()

    def find_triggers(self):
        if self.raw is None:
            return
        if self.settings_changed('annotate'):
            event_annotations = io.get_event_annotations(
                self.raw,
                self.trigger_channel,
                group_window=320
            )
            self.raw_annotated = self.raw.copy().set_annotations(event_annotations)
            event_ids = sorted(set(self.raw_annotated.annotations.description))
            event_config = self.event_config.copy()
            for event_id in event_ids:
                event_config.setdefault(event_id, {'label': event_id, 'visible': True})
            self.event_config = event_config
            self.update_settings('annotate')
        self.update_plot_config()

    def reprocess(self):
        if self.raw_annotated is None:
            return
        if self.settings_changed('filter'):
            self.raw_filtered = self.raw_annotated.copy() \
                .filter(self.filt_lb, self.filt_ub, verbose=False)
            self.update_settings('filter')
        if self.settings_changed('epoch'):
            self.epochs = io.get_epoch_data(
                self.raw_filtered, self.time_lb, self.time_ub, (None, 0)
            )
            self.epochs.load_data()
            self.update_settings('epoch')
        self.update_plots()

    def update_plots(self, event=None):
        if not self.selector.selected:
            return
        if self.epochs is None:
            return
        self.update_plot_config()

        epochs = self.epochs.copy() \
            .set_eeg_reference(self.selector.reference, verbose=False) \
            .pick(self.selector.selected, verbose=False)
        for pc in self.plot_config:
            data = epochs[pc.event].average().get_data(picks=pc.channel)[0]
            pc.plot_item.setData(epochs.times, data)

    def enable_crosshairs(self, pc):
        self.crosshairs.enable(pc)

    def disable_crosshairs(self):
        self.crosshairs.disable()

    def add_analysis(self, pc, label, latency, amplitude):
        items = self.analysis[:]
        item = AnalysisItem(
            channel=pc.channel,
            event_label=pc.event_label,
            label=label,
            latency=latency,
            amplitude=amplitude,
        )
        items.append(item)
        self.analysis = items

    def update_plot_config(self):
        if self.settings_changed('plot_config'):
            channels = set(self.selector.selected)
            events = set(k for k, v in self.event_config.items() if v['visible'])

            # Things have changed. Remove existing plots.
            for config in self.plot_config:
                self.viewbox.removeItem(config.plot_item)

            keys = list(itertools.product(channels, events))
            palette = 'palettable.colorbrewer.qualitative.Dark2_8'
            colors = get_color_cycle(palette, len(keys))

            plot_config = []
            for c, key in zip(colors, keys):
                plot_item = pg.PlotCurveItem(pen=pg.mkPen(c, width=3))
                self.viewbox.addItem(plot_item)
                pc = PlotConfig(
                    presenter=self,
                    channel=key[0],
                    event=key[1],
                    event_label=self.event_config[key[1]]['label'],
                    visible=True,
                    color=Color(*c),
                    plot_item=plot_item,
                )
                plot_config.append(pc)
            self.plot_config = plot_config
            self.update_settings('plot_config')
