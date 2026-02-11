import json
from pathlib import Path

import mne.io
from ncrar_audio import triggers

from .configs import get_config


def get_event_annotations(raw, channel, group_window=320, trig_codes=None,
                          inplace=False):
    if trig_codes is None:
        trig_codes = {}

    if channel.startswith('Status'):
        raise NotImplementedError
        # Something along these lines. Will be in the format Status[c]
        status, time = raw.copy().pick(['Status'])['Status']
        trigs = (status[0].astype('i') >> (c-1)) & 0b1
    else:
        s_trig, time = raw.copy().pick([channel])[channel]
        s_trig = s_trig[0]

    trig_samples = triggers.extract_triggers(s_trig, group_window=group_window)
    onsets = []
    descriptions = []
    for k, v in trig_samples.items():
        onsets.extend(time[v])
        d = trig_codes.get(k, str(k))
        descriptions.extend([d]*len(v))
    return mne.Annotations(
        onset=onsets,
        duration=0.05,
        description=descriptions,
        orig_time=raw.info['meas_date'],
    )


def get_epoch_data(raw, tmin, tmax, baseline):
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        verbose=False,
    )
    return epochs
