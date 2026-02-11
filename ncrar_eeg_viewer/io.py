import json
from pathlib import Path

import mne.io
from ncrar_audio import triggers

from .configs import get_config


def set_annotations(raw, channel, group_window=320, trig_codes=None,
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
    annotation = mne.Annotations(
        onset=onsets,
        duration=0.05,
        description=descriptions,
    )
    if inplace:
        return raw.set_annotations(annotation, verbose=False)
    else:
        return raw.copy().set_annotations(annotation, verbose=False)


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


def auto_preprocess_file(filename):
    filename = Path(filename)
    name, config = get_config(filename)

    raw = mne.io.read_raw_bdf(filename, preload=True)
    raw = set_annotations(raw, **config['trigger'])
    epochs = get_epoch_data(raw, **config['epoch'])

    return epoch_filename


def preprocess():
    import argparse
    parser = argparse.ArgumentParser('ncrar-eeg-preprocess')
    parser.add_argument('filename')
    parser.add_argument('trigger', type=str)
    parser.add_argument('time_lb', default=-2, type=float)
    parser.add_argument('time_ub', default=12, type=float)
    parser.add_argument('--reprocess', action='store_true')
    args = parser.parse_args()
    preprocess_file(args.filename, args.trigger, args.time_lb, args.time_ub,
                    args.reprocess)
