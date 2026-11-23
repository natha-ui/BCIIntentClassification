"""
run_pipeline.py — one entry point for the Track 02 FB-TS pipeline.
Works the same on Windows, macOS and Linux (pure Python, no shell tricks).

  python run_pipeline.py check                         # environment report
  python run_pipeline.py data --bids data/nm000290     # PROTEUS -> EMData/proteus.npz
  python run_pipeline.py train --preset v2             # train + package
  python run_pipeline.py replay runs/v2                # evaluator-style check of the zip
  python run_pipeline.py all --preset v2               # check + train + replay

Presets (all on PROTEUS only, streamed CV = forward-session + leave-one-subject-out,
cell-balanced accuracy, as in docs/RESULTS.md):

  v2              the best known single model (default bands, full window,
                  C=0.05, prior_weight=2, alpha_min=0.03; CV 0.606 / 0.567 on 10-24)
  v2-nocv         same model, no CV: fit + package only (fastest)
  ablations       queue item: recenter on/off x trace_norm on/off x prior_correction 0/0.5
  features-small  queue item (reduced): bands default,physio,wide x windows full,trim, ensemble<=3
  features        queue item (full):    bands default,physio,fbcsp,wide x windows full,trim,multi, ensemble<=3

Use --minutes N to cap the CV time of a grid (configs already scored are kept).
Outputs go to runs/<preset>/ (+ runs/<preset>.zip, + run_log.txt with everything printed).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import json
import os
import platform
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

BASE = "C=0.05,prior_weight=2,alpha_min=0.03"
PRESETS = {
    "v2": dict(set=f"bands=default,windows=full,{BASE}", grid="", ensemble=1, cv=True),
    "v2-nocv": dict(set=f"bands=default,windows=full,{BASE}", grid="", ensemble=1, cv=False),
    "ablations": dict(set=f"bands=default,windows=full,{BASE}",
                      grid="recenter=true,false;trace_norm=true,false;prior_correction=0,0.5",
                      ensemble=1, cv=True),
    "features-small": dict(set=BASE, grid="bands=default,physio,wide;windows=full,trim",
                           ensemble=3, cv=True),
    "features": dict(set=BASE, grid="bands=default,physio,fbcsp,wide;windows=full,trim,multi",
                     ensemble=3, cv=True),
}
DEFAULT_DATA = HERE / "EMData" / "proteus.npz"


class _Tee:
    """Print to the console and to the run log at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            if not st.closed:
                st.write(s)
                st.flush()
        return len(s)

    def flush(self):
        for st in self.streams:
            if not st.closed:
                st.flush()


@contextlib.contextmanager
def logged(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        old = sys.stdout
        sys.stdout = _Tee(old, f)
        try:
            yield
        finally:
            sys.stdout = old


# -----------------------------------------------------------------------------
def cmd_check(args=None):
    """Print (and return) versions of everything the pipeline touches."""
    info = dict(python=platform.python_version(), platform=platform.platform(),
                machine=platform.machine(), cpu_count=os.cpu_count())
    ok = True
    for mod, need in [("numpy", True), ("scipy", True), ("sklearn", True),
                      ("pandas", False), ("mne", False), ("torch", False),
                      ("moabb", False), ("nemar", False), ("benchopt", False)]:
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "installed")
        except Exception:
            info[mod] = None
            if need:
                ok = False
    for k, v in info.items():
        print(f"  {k:10s} {v if v is not None else '-- not installed'}")
    print("  needed for: numpy/scipy/sklearn = everything; mne+pandas = 'data'; "
          "torch = replay with tensors (optional); nemar = PROTEUS download")
    if not ok:
        sys.exit("missing a required package: pip install -r requirements.txt")
    importlib.import_module("submission")   # imports cleanly without the harness
    print("  submission.py imports OK")
    return info


def cmd_data(args):
    out = Path(args.out)
    if args.synthetic:
        import make_synthetic
        make_synthetic.main(["--out", str(out), "--subjects", "5", "--sessions", "3",
                             "--per-class-ctx", "30"])
    elif args.bids:
        import prepare_data
        prepare_data.main(["bids", args.bids, "--out", str(out)])
    elif not out.exists():
        sys.exit(f"{out} does not exist: pass --bids <PROTEUS folder> (see README)")
    import prepare_data
    prepare_data.main(["info", str(out)])


def cmd_train(args):
    import train_fbts
    pr = PRESETS[args.preset]
    data = Path(args.data)
    if not data.exists():
        sys.exit(f"{data} not found: run `python run_pipeline.py data --bids <folder>` first")
    out = Path(args.out) if args.out else HERE / "runs" / args.preset
    argv = [str(data), "--set", pr["set"], "--out", str(out), "-j", str(args.j)]
    if pr["grid"]:
        argv += ["--grid", pr["grid"]]
    if pr["ensemble"] > 1:
        argv += ["--ensemble", str(pr["ensemble"])]
    if not pr["cv"]:
        argv += ["--no-cv"]
    if args.minutes:
        argv += ["--max-minutes", str(args.minutes)]
    if args.j <= 1:
        argv += ["--verbose"]                      # per-fold progress lines
    print(f"preset {args.preset}: python train_fbts.py {' '.join(argv)}")
    t0 = time.time()
    train_fbts.main(argv)
    print(f"total {(time.time() - t0) / 60:.1f} min")
    return out


def _import_fresh(path):
    """Import a submission.py from its folder as an isolated module, like the
    worker does (no reuse of the training-side module)."""
    spec = importlib.util.spec_from_file_location(f"sub_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    old, sys.dont_write_bytecode = sys.dont_write_bytecode, True   # no __pycache__
    try:                                                          # in the folder
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


def cmd_replay(args):
    """Evaluator emulation on the PROTEUS windows: zip layout, load_model(meta),
    one deepcopy + reset_state per session, predict() on (1, 41, 480) tensors,
    output type/shape/range, no writes into the submission folder, latency.
    The accuracy printed is IN-SAMPLE (the model saw these windows): a
    pipeline check, not a score."""
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score
    folder = Path(args.folder)
    zpath = folder.with_suffix(".zip")
    if zpath.exists():
        names = zipfile.ZipFile(zpath).namelist()
        assert "submission.py" in names and all("/" not in n for n in names), names
        print(f"  zip root OK: {names}  ({zpath.stat().st_size / 1e6:.1f} MB)")
    before = {p.name: p.stat().st_mtime for p in folder.iterdir()}
    mod = _import_fresh(folder / "submission.py")
    z = dict(np.load(args.data, allow_pickle=False))   # decompress once
    ch = [str(c) for c in z["ch_names"]]
    meta = dict(submission_dir=folder, device="cpu", n_chans=len(ch),
                n_times=z["X"].shape[-1], sfreq=float(z["sfreq"]), ch_names=ch,
                n_classes=3)
    model = mod.Solver().load_model(meta)
    try:
        import torch
        to_in = lambda x: torch.from_numpy(x)  # noqa: E731
        print("  inputs: torch tensors (as on the worker)")
    except Exception:
        to_in = lambda x: x  # noqa: E731
        print("  inputs: numpy arrays (torch not installed)")
    groups = np.array([f"{a}/{b}" for a, b in zip(z["subject"], z["session"])])
    sessions = list(dict.fromkeys(groups))[: args.sessions or None]
    pred, idx_all, n, t_pred = [], [], 0, 0.0
    for g in sessions:
        m = copy.deepcopy(model)
        if hasattr(m, "reset_state"):
            m.reset_state()
        idx = np.flatnonzero(groups == g)
        for i in idx:
            x = to_in(np.ascontiguousarray(z["X"][i:i + 1], dtype=np.float32))
            t0 = time.perf_counter()
            out = m.predict(x)
            t_pred += time.perf_counter() - t0
            out = np.asarray(out.detach().cpu() if hasattr(out, "detach") else out)
            assert out.shape == (1,), out.shape
            assert np.issubdtype(out.dtype, np.integer) and 0 <= out[0] <= 2, out
            pred.append(int(out[0]))
        idx_all += list(idx)
        n += len(idx)
    after = {p.name: p.stat().st_mtime for p in folder.iterdir()}
    assert before == after, "the model wrote into its submission folder"
    idx_all, pred = np.array(idx_all), np.array(pred)
    y = z["y"][idx_all]
    cells = np.array([f"{g}|{c}" for g, c in zip(groups[idx_all], z["context"][idx_all])])
    bacc = np.mean([balanced_accuracy_score(y[cells == c], pred[cells == c])
                    for c in np.unique(cells)])
    ms = 1e3 * t_pred / n
    print(f"  streamed {n} windows over {len(sessions)} session(s): output (1,) int in "
          f"{{0,1,2}} OK, deepcopy/reset_state OK, folder untouched OK")
    print(f"  latency {ms:.1f} ms/window -> ~{ms * 12000 / 6e4:.0f} min for ~12k windows "
          f"(worker limit 60 min)")
    print(f"  in-sample cell-balanced accuracy {bacc:.3f} (training data: pipeline "
          f"check only, NOT a score)")


def cmd_all(args):
    out = Path(args.out) if args.out else HERE / "runs" / args.preset
    with logged(out.with_name(out.name + "_run_log.txt")):
        print(f"== {time.strftime('%Y-%m-%d %H:%M')}  preset {args.preset}")
        print("== check");  env = cmd_check()
        print("== train");  out = cmd_train(args)
        (Path(out) / "environment.json").write_text(json.dumps(env, indent=2))
        print("== replay"); args.folder = str(out); cmd_replay(args)
        print(f"== done: {out}  ({Path(out).with_suffix('.zip')})")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(func=cmd_check)

    d = sub.add_parser("data", help="build EMData/proteus.npz")
    d.add_argument("--bids", help="PROTEUS BIDS folder (nm000290)")
    d.add_argument("--synthetic", action="store_true", help="fake data, for a dry run")
    d.add_argument("--out", default=str(DEFAULT_DATA))
    d.set_defaults(func=cmd_data)

    for name, fn in (("train", cmd_train), ("all", cmd_all)):
        t = sub.add_parser(name)
        t.add_argument("--preset", choices=PRESETS, default="v2")
        t.add_argument("--data", default=str(DEFAULT_DATA))
        t.add_argument("--out", default="")
        t.add_argument("-j", type=int, default=1, help="parallel grid workers")
        t.add_argument("--minutes", type=float, default=0, help="CV time cap for grids")
        t.add_argument("--sessions", type=int, default=3, help="sessions to replay (0 = all)")
        t.set_defaults(func=fn)

    r = sub.add_parser("replay")
    r.add_argument("folder", help="e.g. runs/v2")
    r.add_argument("--data", default=str(DEFAULT_DATA))
    r.add_argument("--sessions", type=int, default=3, help="0 = all sessions")
    r.set_defaults(func=cmd_replay)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
