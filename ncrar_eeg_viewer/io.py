from pathlib import Path

import mne.io
from ncrar_audio import triggers


def set_annotations(raw, time, s):
    trig_samples = triggers.extract_triggers(s, group_window=320)
    pos_polarity = trig_samples[1]
    neg_polarity = trig_samples[2]
    pos_timestamps = time[pos_polarity]
    neg_timestamps = time[neg_polarity]

    # Convert into an annotations object
    annotations = mne.Annotations(
        onset=pos_timestamps,
        duration=0.05,
        description="1",
    )

    annotations.append(
        onset=neg_timestamps,
        duration= 0.05,
        description="2",
    )

    return raw.set_annotations(annotations)


def get_epoch_data(raw, time_lb=-2e-3, time_ub=14e-3):
    events, event_id = mne.events_from_annotations(raw, event_id={'1': 0, '2': 1})
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=time_lb,
        tmax=time_ub,
        #baseline=(None, 0), # default: 'none'=beginning of data until time point 0
        #reject={'eeg': 40e-6}, #peak-to-peak (20 microvolt peak reject)
    )
    return epochs


def preprocess_file(filename, trigger, time_lb, time_ub, reprocess=False):
    filename = Path(filename)
    epoch_filename = filename.parent / f'{filename.stem}_{trigger}_{time_lb}_to_{time_ub}ms-epo.fif'
    if epoch_filename.exists() and not reprocess:
        return epoch_filename

    raw = mne.io.read_raw_bdf(filename, preload=True)

    trigger = trigger.lower()
    if trigger == 'erg2':
        trigs, time = raw.copy().pick(['Erg2'])['Erg2']
        trigs = trigs[0]
    elif trigger == 'status[9]':
        status, time = raw.copy().pick(['Status'])['Status']
        trigs = (status[0].astype('i') >> 8) & 0b1

    set_annotations(raw, time, trigs)
    epochs = get_epoch_data(raw, time_lb*1e-3, time_ub*1e-3)
    epochs.save(epoch_filename, overwrite=True)
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
