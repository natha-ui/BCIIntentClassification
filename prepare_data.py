"""
prepare_data.py — turn any dataset into the window format train_fbts.py
consumes, applying the SAME fixed preprocessing the competition applies
before streaming (so training features match what predict() receives):

    notch 50 & 60 Hz -> band-pass 0.1-75 Hz (at the native rate)
    -> resample to 120 Hz -> 4 s window -> microvolts

Output: one .npz per dataset with
    X        float32 (n, C, 480)   windows at 120 Hz, in µV
    y        int64   (n,)          0 = motor imagery, 1 = mental subtraction,
                                   2 = word generation
    subject  str (n,), session str (n,), context str (n,)
    ch_names str (C,), sfreq float, dataset str
Windows of each (subject, session) are stored in recording order.

Sub-commands
------------
  bids       PROTEUS (NEMAR nm000290) straight from the extracted/downloaded
             BIDS folder:
               python prepare_data.py bids data/nm000290 --out EMData/proteus.npz
             contexts are collapsed to graz / brainhero (the official cells).

  scherer2015  BNCI 2015-004 (Scherer et al. 2015) via MOABB: 9 subjects x
             2 sessions, 30 ch, word association / mental subtraction /
             right-hand MI (+ navigation, feet: dropped).  Class names are
             read from each downloaded .mat (MOABB's own event labels for
             this set are misleading).  Cue at +3 s, window cue+1..cue+5.
               python prepare_data.py scherer2015 --out EMData/scherer2015.npz

  dreyer2023 Dreyer et al. 2023 (MOABB Dreyer2023, the same Bordeaux Graz
             protocol), 27 ch, left/right-hand MI -> class 0 ONLY.  87
             subjects / ~19 GB: take a subset with --subjects.  Window
             offset 0 s after the MOABB event, as NeuralBench's dreyer2023
             config.
               python prepare_data.py dreyer2023 --subjects 1-20 --out EMData/dreyer2023.npz

  stieger2021  Stieger 2021 (MOABB): hand MI -> 0, rest -> unlabelled (-1).
             ~10 GB per subject; default --subjects 1-5.

  zyma2019   Zyma 2019 (PhysioNet eegmat): serial subtraction -> 1, rest ->
             unlabelled; 4-s non-overlapping windows from the continuous files.
               python prepare_data.py zyma2019 data/eegmat --out EMData/zyma2019.npz

  Unlabelled windows (y = -1) shape their session's re-centring mean but are
  never used to train the classifier.

  moabb      any other MOABB dataset with an explicit class map:
               python prepare_data.py moabb --dataset BNCI2014_001 --class-map "left_hand:0,right_hand:0" --win-start 1.0 --out EMData/bnci2a.npz

  info       print the summary of one or more .npz files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

TARGET_SFREQ = 120.0
WIN_LEN = 4.0
N_TIMES = int(round(WIN_LEN * TARGET_SFREQ))
PROTEUS_CODES = {"mi": 0, "sub": 1, "word": 2}
EEG_EXTS = (".edf", ".bdf", ".fif", ".set", ".vhdr")


def save_npz(out, X, y, subject, session, context, ch_names, dataset):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    assert X.ndim == 3 and X.shape[-1] == N_TIMES, X.shape
    np.savez_compressed(
        out, X=X, y=y, subject=np.asarray(subject, dtype=str),
        session=np.asarray(session, dtype=str),
        context=np.asarray(context, dtype=str),
        ch_names=np.asarray(ch_names, dtype=str),
        sfreq=np.float64(TARGET_SFREQ), dataset=str(dataset))
    n_sess = len(np.unique([f"{s}/{t}" for s, t in zip(subject, session)]))
    print(f"wrote {out}: X{X.shape}, classes {np.bincount(y[y >= 0], minlength=3)}"
          f" + {int((y < 0).sum())} unlabelled, "
          f"{len(np.unique(subject))} subjects, {n_sess} sessions")


def parse_class_map(s):
    out = {}
    for item in s.split(","):
        k, v = item.rsplit(":", 1)
        out[k.strip()] = int(v)
    return out


def parse_subjects(s, available):
    """'' -> all; '1-20' / '1,3,5' / '1-5,9' -> list of ints."""
    if not s:
        return list(available)
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [x for x in out if x in set(available)]


# -----------------------------------------------------------------------------
# contract preprocessing + windowing
# -----------------------------------------------------------------------------
def contract_preprocess(raw):
    """EEG only; notch 50/60, band-pass 0.1-75 Hz, resample 120 Hz."""
    raw = raw.copy().load_data()
    raw.pick("eeg")
    sf = raw.info["sfreq"]
    notch = [f for f in (50.0, 60.0) if f < sf / 2 - 1]
    if notch:
        raw.notch_filter(notch, verbose=False)
    raw.filter(0.1, min(75.0, 0.45 * sf), verbose=False)
    raw.resample(TARGET_SFREQ, verbose=False)
    return raw


def windows_at(raw, onsets_s, win_start):
    """Cut (n, C, 480) µV windows starting `win_start` s after each onset
    (onsets in seconds from the start of `raw`).  Windows that do not fit
    are dropped; returns (X, kept_index)."""
    data = raw.get_data(units="uV")
    sf = raw.info["sfreq"]
    starts = np.round((np.asarray(onsets_s) + win_start) * sf).astype(int)
    keep = np.flatnonzero((starts >= 0) & (starts + N_TIMES <= data.shape[1]))
    X = np.stack([data[:, s:s + N_TIMES] for s in starts[keep]]) if len(keep) \
        else np.zeros((0, data.shape[0], N_TIMES))
    return X.astype(np.float32), keep


def raw_events(raw, name_of_code=None, wanted=None):
    """(onset_seconds, event_name) of a raw's events: annotations if any of
    them is in `wanted`, else the stim channel (codes -> `name_of_code`)."""
    import mne
    ann = raw.annotations
    if len(ann):
        names = np.array([str(d) for d in ann.description], dtype=object).astype("U64")
        if wanted is None or np.isin(names, list(wanted)).any():
            return np.asarray(ann.onset) - raw.first_time, names
    stim = mne.pick_types(raw.info, stim=True)
    if len(stim) == 0:
        return np.zeros(0), np.zeros(0, dtype=str)
    ev = mne.find_events(raw, stim_channel=raw.ch_names[stim[0]],
                         shortest_event=1, verbose=False)
    on = (ev[:, 0] - raw.first_samp) / raw.info["sfreq"]
    names = [(name_of_code or {}).get(int(c), str(int(c))) for c in ev[:, 2]]
    return on, np.asarray(names, dtype=str)


class _Collector:
    def __init__(self):
        self.X, self.y, self.subj, self.sess, self.ctx = [], [], [], [], []
        self.ch_names = None

    def add(self, raw, onsets, labels, subj, sess, ctx, win_start):
        if self.ch_names is None:
            self.ch_names = list(raw.ch_names)
        elif list(raw.ch_names) != self.ch_names:
            raw.reorder_channels(self.ch_names)
        Xw, keep = windows_at(raw, onsets, win_start)
        if not len(keep):
            return 0
        self.X.append(Xw)
        self.y += [int(v) for v in np.asarray(labels)[keep]]
        self.subj += [subj] * len(keep)
        self.sess += [sess] * len(keep)
        self.ctx += ([ctx] * len(keep) if isinstance(ctx, str)
                     else list(np.asarray(ctx)[keep]))
        return len(keep)

    def save(self, out, name):
        if not self.X:
            sys.exit("no windows extracted — check the class map / event names")
        save_npz(out, np.concatenate(self.X), self.y, self.subj, self.sess,
                 self.ctx, self.ch_names, name)


# -----------------------------------------------------------------------------
# PROTEUS / any EEG-BIDS cue dataset, straight from the extracted folder
# -----------------------------------------------------------------------------
def _bids_entities(name):
    return dict(p.split("-", 1) for p in Path(name).name.split("_") if "-" in p)


def proteus_context(task):
    t = task.lower()
    return "graz" if "graz" in t else ("brainhero" if ("bh" in t or "hero" in t) else t)


def cmd_bids(args):
    import mne
    import pandas as pd
    root = Path(args.root).expanduser().resolve()
    files = [f for f in root.rglob("*_eeg.*") if f.suffix.lower() in EEG_EXTS]

    def order(f):  # recording order: subject, session, run number
        e = _bids_entities(f)
        r = e.get("run", "0")
        return (e.get("sub", ""), e.get("ses", ""), int(r) if r.isdigit() else 0, f.name)

    files = sorted(files, key=order)
    if not files:
        sys.exit(f"no *_eeg.edf/.bdf/.fif/.set/.vhdr found under {root} — "
                 "is the dataset downloaded / the zip extracted?")
    codes = parse_class_map(args.class_map) if args.class_map else PROTEUS_CODES
    col = _Collector()
    for f in files:
        tsv = Path(str(f).rsplit("_eeg.", 1)[0] + "_events.tsv")
        if not tsv.exists():
            continue
        ev = pd.read_csv(tsv, sep="\t")
        ev = ev[ev["trial_type"].isin(codes)].sort_values("onset")
        if ev.empty:                                   # baseline runs
            continue
        raw = contract_preprocess(mne.io.read_raw(f, preload=True, verbose=False))
        e = _bids_entities(f)
        n = col.add(raw, ev["onset"].to_numpy(), ev["trial_type"].map(codes).to_numpy(),
                    f"{args.name}/sub-{e.get('sub', '0')}", f"ses-{e.get('ses', '0')}",
                    proteus_context(e.get("task", "0")), args.win_start)
        print(f"  {f.name}: {n} windows", flush=True)
    col.save(args.out, args.name)


# -----------------------------------------------------------------------------
# MOABB datasets
# -----------------------------------------------------------------------------
def _moabb_loop(dataset, subjects, class_of_event, name, win_start, col,
                name_of_code_fn=None):
    for s in subjects:
        data = dataset.get_data(subjects=[s])[s]
        name_of_code = name_of_code_fn(s) if name_of_code_fn else \
            {v: k for k, v in dataset.event_id.items()}
        for ses_name, runs in data.items():
            # runs of a session in recording order
            for run_name in sorted(runs, key=lambda r: (len(str(r)), str(r))):
                raw = runs[run_name]
                if raw is None:
                    continue
                on, names = raw_events(raw, name_of_code, class_of_event)
                keep = np.array([n in class_of_event for n in names], bool)
                if not keep.any():
                    continue
                on, names = on[keep], names[keep]
                order = np.argsort(on, kind="stable")
                on, names = on[order], names[order]
                labels = np.array([class_of_event[n] for n in names])
                prep = contract_preprocess(raw)
                n = col.add(prep, on, labels, f"{name}/{s}", f"ses-{ses_name}",
                            str(run_name), win_start)
                print(f"  subject {s} session {ses_name} run {run_name}: {n} windows",
                      flush=True)


def cmd_moabb(args):
    import moabb.datasets as mds
    dataset = getattr(mds, args.dataset)()
    col = _Collector()
    _moabb_loop(dataset, parse_subjects(args.subjects, dataset.subject_list),
                parse_class_map(args.class_map), args.dataset, args.win_start, col)
    col.save(args.out, args.dataset)


# Scherer 2015 class names as stored in the BNCI .mat files (run.classes)
SCHERER_MAP = {"word": 2, "word association": 2, "word_ass": 2,
               "sub": 1, "subtraction": 1, "mental subtraction": 1,
               "hand": 0, "right hand": 0, "right_hand": 0}


def scherer_class_names(dataset, subject):
    """{code: class name} read from the subject's downloaded .mat file."""
    from scipy.io import loadmat
    files = dataset.data_path(subject)
    files = files if isinstance(files, (list, tuple)) else [files]
    data = loadmat(str(files[0]), struct_as_record=False, squeeze_me=True)["data"]
    runs = data if isinstance(data, np.ndarray) else [data]
    for run in runs:
        classes = np.atleast_1d(run.classes)
        if len(classes):
            return {i + 1: str(c).strip().lower() for i, c in enumerate(classes)}
    raise RuntimeError("no class names in the .mat file")


def cmd_scherer2015(args):
    import moabb.datasets as mds
    dataset = mds.BNCI2015_004()
    subjects = parse_subjects(args.subjects, dataset.subject_list)
    names = scherer_class_names(dataset, subjects[0])
    print(f"class names in the .mat: {names}")
    mapped = {n: SCHERER_MAP[n] for n in names.values() if n in SCHERER_MAP}
    if sorted(mapped.values()) != [0, 1, 2]:
        sys.exit(f"could not find hand / subtraction / word in {names}; "
                 "edit SCHERER_MAP")
    print(f"using {mapped} (others dropped)")
    col = _Collector()
    _moabb_loop(dataset, subjects, mapped, "scherer2015", args.win_start, col,
                name_of_code_fn=lambda s: scherer_class_names(dataset, s))
    col.save(args.out, "scherer2015")


def cmd_dreyer2023(args):
    import moabb.datasets as mds
    dataset = mds.Dreyer2023()
    hands = {"left": {"left_hand": 0}, "right": {"right_hand": 0},
             "both": {"left_hand": 0, "right_hand": 0}}[args.hands]
    col = _Collector()
    _moabb_loop(dataset, parse_subjects(args.subjects, dataset.subject_list),
                hands, "dreyer2023", args.win_start, col)
    col.save(args.out, "dreyer2023")


def cmd_stieger2021(args):
    """Stieger 2021 (MOABB): left/right/both-hand MI -> 0, rest -> -1
    (unlabelled: shapes the session's re-centring mean only)."""
    import moabb.datasets as mds
    dataset = mds.Stieger2021()
    class_map = {"left_hand": 0, "right_hand": 0, "both_hand": 0,
                 "rest": -1 if args.rest == "unlabelled" else None}
    class_map = {k: v for k, v in class_map.items() if v is not None}
    col = _Collector()
    _moabb_loop(dataset, parse_subjects(args.subjects, dataset.subject_list),
                class_map, "stieger2021", args.win_start, col)
    col.save(args.out, "stieger2021")


def _clean_channel_names(raw):
    """'EEG Fp1' / 'Fp1-LE' / 'FP1' -> 'Fp1'; keep only 10-05 EEG positions."""
    import warnings
    import mne
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mont = mne.channels.make_standard_montage("colin27_1005")
        except Exception:
            mont = mne.channels.make_standard_montage("standard_1005")
    std = {n.lower(): n for n in mont.ch_names}
    alias = {"t3": "T7", "t4": "T8", "t5": "P7", "t6": "P8"}
    rename, keep = {}, []
    for ch in raw.ch_names:
        n = ch.replace("EEG", "").strip().split("-")[0].strip().lower()
        new = alias.get(n) or std.get(n)
        if n in ("a1", "a2", "m1", "m2"):         # reference electrodes
            new = None
        if new and new not in rename.values():
            rename[ch] = new
            keep.append(new)
    raw.rename_channels(rename)
    raw.pick(keep)
    raw.set_channel_types({c: "eeg" for c in keep})
    return raw


def cmd_zyma2019(args):
    """Zyma 2019 / PhysioNet 'eegmat': SubjectXX_2.edf = serial subtraction
    (class 1), SubjectXX_1.edf = rest (unlabelled by default).  Continuous
    recordings are cut into non-overlapping 4-s windows; one session per
    subject = rest windows followed by arithmetic windows."""
    import mne
    root = Path(args.root).expanduser().resolve()
    files = sorted(root.rglob("Subject*_2.edf"))
    if not files:
        sys.exit(f"no Subject*_2.edf under {root} (PhysioNet eegmat files)")
    col = _Collector()
    for f2 in files:
        subj = f2.name.split("_")[0]
        parts = []
        f1 = f2.with_name(f2.name.replace("_2.edf", "_1.edf"))
        if f1.exists() and args.rest == "unlabelled":
            parts.append((f1, -1))
        parts.append((f2, 1))
        for f, label in parts:
            raw = _clean_channel_names(mne.io.read_raw_edf(f, preload=True, verbose=False))
            raw = contract_preprocess(raw)
            dur = raw.n_times / raw.info["sfreq"]
            onsets = np.arange(0.0, dur - WIN_LEN + 1e-9, WIN_LEN)
            if label < 0:                       # last 60 s of rest only, so
                onsets = onsets[-15:]           # rest ~ arithmetic in the mean
            n = col.add(raw, onsets, np.full(len(onsets), label),
                        f"zyma2019/{subj}", "ses-1", "rest" if label < 0 else "arith", 0.0)
            print(f"  {f.name}: {n} windows ({'rest, unlabelled' if label < 0 else 'subtraction'})")
    col.save(args.out, "zyma2019")


def cmd_info(args):
    for f in args.files:
        z = np.load(f, allow_pickle=False)
        sessions = np.unique([f"{s}/{t}" for s, t in zip(z["subject"], z["session"])])
        print(f"{f}: {z['dataset']} X{z['X'].shape} @ {float(z['sfreq'])} Hz, "
              f"classes {np.bincount(z['y'][z['y'] >= 0], minlength=3)} + "
              f"{int((z['y'] < 0).sum())} unlabelled, "
              f"{len(np.unique(z['subject']))} subjects, {len(sessions)} sessions, "
              f"contexts {sorted(set(z['context'].tolist()))[:4]}, "
              f"channels {[str(c) for c in z['ch_names'][:6]]}...")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("bids", help="PROTEUS / EEG-BIDS folder")
    d.add_argument("root")
    d.add_argument("--out", required=True)
    d.add_argument("--name", default="proteus")
    d.add_argument("--class-map", default="", help='default "mi:0,sub:1,word:2"')
    d.add_argument("--win-start", type=float, default=1.0, help="s after cue")
    d.set_defaults(func=cmd_bids)

    s = sub.add_parser("scherer2015", help="BNCI 2015-004 via MOABB")
    s.add_argument("--out", required=True)
    s.add_argument("--subjects", default="")
    s.add_argument("--win-start", type=float, default=4.0,
                   help="s after the trial marker (cue at +3 s)")
    s.set_defaults(func=cmd_scherer2015)

    r = sub.add_parser("dreyer2023", help="Dreyer 2023 MI via MOABB")
    r.add_argument("--out", required=True)
    r.add_argument("--subjects", default="1-20", help='e.g. "1-20", "" = all 87')
    r.add_argument("--hands", choices=["left", "right", "both"], default="both")
    r.add_argument("--win-start", type=float, default=0.0,
                   help="s after the MOABB event (NeuralBench uses 0)")
    r.set_defaults(func=cmd_dreyer2023)

    t = sub.add_parser("stieger2021", help="Stieger 2021 MI via MOABB (~10 GB/subject)")
    t.add_argument("--out", required=True)
    t.add_argument("--subjects", default="1-5", help='e.g. "1-5"; "" = all 62 (~640 GB)')
    t.add_argument("--rest", choices=["unlabelled", "drop"], default="unlabelled")
    t.add_argument("--win-start", type=float, default=1.0, help="s after the target cue")
    t.set_defaults(func=cmd_stieger2021)

    zy = sub.add_parser("zyma2019", help="Zyma 2019 mental arithmetic (PhysioNet eegmat folder)")
    zy.add_argument("root", help="folder containing Subject00_1.edf, Subject00_2.edf, ...")
    zy.add_argument("--out", required=True)
    zy.add_argument("--rest", choices=["unlabelled", "drop"], default="unlabelled")
    zy.set_defaults(func=cmd_zyma2019)

    b = sub.add_parser("moabb", help="any MOABB dataset")
    b.add_argument("--dataset", required=True)
    b.add_argument("--class-map", required=True)
    b.add_argument("--subjects", default="")
    b.add_argument("--win-start", type=float, default=1.0)
    b.add_argument("--out", required=True)
    b.set_defaults(func=cmd_moabb)

    c = sub.add_parser("info")
    c.add_argument("files", nargs="+")
    c.set_defaults(func=cmd_info)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
