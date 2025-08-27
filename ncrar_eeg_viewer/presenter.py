from atom.api import Atom, Int, Value, Typed

from matplotlib.figure import Figure
import mne.io
from scipy import signal

from biosemi_enaml.electrode_selector import BiosemiElectrodeSelector


class Presenter(Atom):

    selector = Typed(BiosemiElectrodeSelector)
    figure = Typed(Figure)
    axes = Value()
    eeg_plot = Value()

    filename = Value()
    epochs = Value()

    filt_lb = Int(100)
    filt_ub = Int(1500)

    def __init__(self):
        self.selector = BiosemiElectrodeSelector(n_channels=64,
                                                 include_exg=False,
                                                 select_mode='single')
        self.selector.observe('reference', self._update_plot)
        self.selector.observe('selected', self._update_plot)

        self.figure = Figure(facecolor='#f3f3f3')
        self.axes = self.figure.add_axes([0.15, 0.10, 0.75, 0.8])
        self.axes.set_ylabel('Signal (V)')
        self.axes.set_xlabel('Time (s)')
        self.eeg_plot, = self.axes.plot([], [], 'k-')

    def load_file(self, filename):
        self.filename = filename
        self.epochs = mne.read_epochs(filename)
        extra = [c for c in self.epochs.info.ch_names \
                 if c not in set(self.selector.coords.index.values)]
        for ch in ('Status', 'Erg1', 'Erg2'):
            if ch in extra:
                extra.remove(ch)
        self.selector.extra = extra

    def set_channels(self, channels):
        if isinstance(channels, str):
            channels = [channels]
        self.selector.selected = channels

    def set_reference_channels(self, channels):
        if isinstance(channels, str):
            channels = {channels}
        self.selector.reference = set(channels)

    def _observe_filt_lb(self, event):
        self._update_plot()

    def _observe_filt_ub(self, event):
        self._update_plot()

    def _update_plot(self, event=None):
        if not self.selector.selected:
            return
        data = self.epochs.copy() \
            .set_eeg_reference(self.selector.reference) \
            .pick(self.selector.selected) \
            .get_data()[:, 0]

        b, a = signal.iirfilter(4, [self.filt_lb, self.filt_ub],
                                fs=self.epochs.info['sfreq'])
        data = signal.filtfilt(b, a, data, axis=-1)
        x = self.epochs.times
        y = data.mean(axis=0)
        self.eeg_plot.set_data((x, y))
        self.axes.relim()
        self.axes.autoscale_view(True,True,True)
        self.figure.canvas.draw()
