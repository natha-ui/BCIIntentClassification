"""
train_fbts.py — cross-validate, ensemble and package the FB-TS decoder.

Input: .npz window files from prepare_data.py.  The FIRST file is the
primary dataset (PROTEUS): only its windows are ever used as test data,
and its montage is the one the submission receives.  Further files
(Scherer2015, Dreyer2023, ...) are extra TRAINING data only.

Cross-validation (on the primary dataset, streamed exactly like Codabench:
reset_state per session, windows in time order, cell-balanced accuracy):
  forward : per subject with >= 2 sessions, train on everyone else + its
            earlier sessions, test each later session  (cross-session)
  loso    : leave one subject out                       (unseen participant)
  both    : mean of the two
Every score line also reports `rho`: the mean over test sessions of the
largest predicted-class fraction (~0.33 = balanced predictions, near 1 =
collapse onto one class), the free collapse diagnostic from the digests.

Configuration keys (for --set / --grid):
  model   : bands (default|fbcsp|repo|physio|wide|fine), windows
            (full|trim|early|late|halves|multi), C, prior_weight, alpha_min,
            train_mode (batch|online), trace_norm, recenter,
            prior_correction, order
  trainer : data      primary | all     use the extra files or not (default all)
            channels  target | common   target = interpolate extra corpora onto
                                        the primary montage; common = keep only
                                        channels every file has (no
                                        interpolation; the member then selects
                                        them from the 41 at test time)
            w_extra   sample weight of extra-corpus windows (default 1.0)

Ensembling: every grid configuration's out-of-fold probabilities are kept,
and --ensemble K greedily picks up to K members (with replacement) that
maximise the averaged-probability CV score.  No retraining is needed to
score an ensemble; only the chosen members are refit on all data.

Runs on Linux, macOS and Windows.  -j N uses worker processes where the OS
can fork (Linux/macOS) and threads elsewhere (Windows).  --max-minutes stops
the grid early and keeps the configurations already scored.

Examples (one line each, so they paste into PowerShell too)
--------
  python train_fbts.py EMData/proteus.npz --cv both --set "C=0.05,prior_weight=2,alpha_min=0.03" --grid "bands=default,physio,fbcsp,wide;windows=full,multi" --ensemble 3 --out runs/ens

Output: <out>/submission.py + fbts_weights.npz + training_info.json +
cv_results.json, and <out>.zip (submission.py + weights at the zip root).
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from submission import (FBTSEnsemble, FBTSModel, WEIGHTS_FILE,  # noqa: E402
                        common_channels, montage_matrix, _norm)

TRAINER_KEYS = ("data", "channels", "w_extra")
FEATURE_KEYS = ("bands", "windows", "trace_norm", "order")
SEALED_WINDOWS_GUESS = 12000      # rough size of the sealed test, for timing


# -----------------------------------------------------------------------------
# data
# -----------------------------------------------------------------------------
class Corpus:
    """All files, plus channel-mode views and a covariance cache."""

    def __init__(self, files):
        self.files = [np.load(f, allow_pickle=False) for f in files]
        self.names = [str(z["dataset"]) for z in self.files]
        sf = {float(z["sfreq"]) for z in self.files}
        if len(sf) != 1:
            sys.exit(f"all files must share one sampling rate, got {sf}")
        self.sfreq = sf.pop()
        self.ch = [[str(c) for c in z["ch_names"]] for z in self.files]
        y, subj, sess, ctx, src = [], [], [], [], []
        for i, z in enumerate(self.files):
            y.append(z["y"]); subj += list(z["subject"]); sess += list(z["session"])
            ctx += list(z["context"]); src += [i] * len(z["y"])
        self.y = np.concatenate(y)
        self.subject, self.session = np.asarray(subj), np.asarray(sess)
        self.context, self.src = np.asarray(ctx), np.asarray(src)
        self.groups = np.array([f"{a}/{b}" for a, b in zip(self.subject, self.session)])
        self.primary = self.src == 0
        self._X, self._covs = {}, {}
        for i, z in enumerate(self.files):
            print(f"  [{i}] {self.names[i]}: {len(z['y'])} windows, "
                  f"{len(self.ch[i])} ch, classes {np.bincount(z['y'][z['y'] >= 0], minlength=3)}"
                  f" + {int((z['y'] < 0).sum())} unlabelled"
                  + ("  (primary: CV test data)" if i == 0 else "  (extra training data)"))
        if len(self.files) > 1:
            print(f"  common channels across files: {len(self.common_names)}")

    @property
    def common_names(self):
        return common_channels(self.ch)

    def montage(self, mode):
        return self.ch[0] if (mode == "target" or len(self.files) == 1) else self.common_names

    def X(self, mode):
        mode = "target" if len(self.files) == 1 else mode
        if mode not in self._X:
            dst = self.montage(mode)
            parts = []
            for names, z in zip(self.ch, self.files):
                Xf = z["X"].astype(np.float32)
                if [_norm(c) for c in names] != [_norm(c) for c in dst]:
                    A = montage_matrix(names, dst).astype(np.float32)
                    Xf = np.einsum("dc,nct->ndt", A, Xf)
                parts.append(Xf)
            self._X[mode] = np.concatenate(parts)
        return self._X[mode]

    def covs(self, mode, feat):
        mode = "target" if len(self.files) == 1 else mode
        key = (mode, json.dumps(feat, sort_keys=True))
        if key not in self._covs:
            t0 = time.time()
            m = FBTSModel(sfreq=self.sfreq, **feat)
            X = self.X(mode)
            self._covs[key] = np.concatenate(
                [m.covariances(X[i:i + 256], dtype=np.float32)
                 for i in range(0, len(X), 256)], axis=1)
            print(f"  covariances {mode} {feat}: {self._covs[key].shape[0]} blocks, "
                  f"{time.time() - t0:.0f}s", flush=True)
        return self._covs[key]


def split_cfg(cfg):
    trainer = dict(data="all", channels="target", w_extra=1.0)
    trainer.update({k: v for k, v in cfg.items() if k in TRAINER_KEYS})
    model = {k: v for k, v in cfg.items() if k not in TRAINER_KEYS}
    feat = {k: model[k] for k in FEATURE_KEYS if k in model}
    return trainer, model, feat


# -----------------------------------------------------------------------------
# CV
# -----------------------------------------------------------------------------
def make_folds(cp, cv):
    """(train_mask_over_primary, test_mask_over_primary, kind, name)."""
    subj, sess = cp.subject, cp.session
    P = cp.primary
    folds = []
    if cv in ("forward", "both"):
        for s in np.unique(subj[P]):
            order = np.unique(sess[P & (subj == s)])
            if len(order) < 2:
                continue
            for i in range(1, len(order)):
                te = P & (subj == s) & (sess == order[i])
                tr = P & ((subj != s) | ((subj == s) & np.isin(sess, order[:i])))
                folds.append((tr, te, "forward", f"{s} -> {order[i]}"))
    if cv in ("loso", "both"):
        for s in np.unique(subj[P]):
            folds.append((P & (subj != s), P & (subj == s), "loso", f"{s}"))
    return folds


def cells_of(cp, idx):
    return np.array([f"{g}|{c}" for g, c in zip(cp.groups[idx], cp.context[idx])])


def cell_bacc(y, pred, cells):
    return float(np.mean([balanced_accuracy_score(y[cells == c], pred[cells == c])
                          for c in np.unique(cells)]))


def stream_proba(model, covs, groups):
    """Probabilities for test windows, session by session, causally."""
    P = np.empty((covs.shape[1], len(model.classes_)))
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        model.reset_state()
        P[idx] = model.predict_proba_covs(covs[:, idx])
    return P


def fit_member(cp, cfg, train_mask):
    trainer, model_kw, feat = split_cfg(cfg)
    mask = train_mask.copy()
    if trainer["data"] == "all":
        mask |= ~cp.primary
    idx = np.flatnonzero(mask)
    covs = cp.covs(trainer["channels"], feat)
    sw = np.where(cp.primary[idx], 1.0, float(trainer["w_extra"]))
    m = FBTSModel(sfreq=cp.sfreq, **model_kw)
    m.fit(None, cp.y[idx], cp.groups[idx], ch_names=cp.montage(trainer["channels"]),
          covs=covs[:, idx], sample_weight=sw if (sw != 1).any() else None)
    return m, covs


def evaluate(cp, cfg, folds, verbose=False):
    """Returns ({kind: score, 'mean': ..., 'rho': ...}, [probs per fold])."""
    probs = []
    for k, (tr, te, kind, name) in enumerate(folds):
        t0 = time.time()
        m, covs = fit_member(cp, cfg, tr)
        te_idx = np.flatnonzero(te)
        P = stream_proba(m, covs[:, te_idx], cp.groups[te_idx])
        probs.append(P)
        if verbose:
            sc = cell_bacc(cp.y[te_idx], m.classes_[P.argmax(1)], cells_of(cp, te_idx))
            print(f"    fold {k + 1}/{len(folds)} {kind} {name}: {sc:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return score_probs(cp, folds, probs, classes=m.classes_), probs


def score_probs(cp, folds, probs, classes=np.arange(3)):
    per = {"forward": [], "loso": []}
    rho = []
    for (tr, te, kind, _), P in zip(folds, probs):
        idx = np.flatnonzero(te)
        pred = classes[P.argmax(1)]
        per[kind].append(cell_bacc(cp.y[idx], pred, cells_of(cp, idx)))
        for g in np.unique(cp.groups[idx]):          # collapse diagnostic
            pg = pred[cp.groups[idx] == g]
            rho.append(np.bincount(pg, minlength=len(classes)).max() / len(pg))
    out = {k: round(float(np.mean(v)), 4) for k, v in per.items() if v}
    out["mean"] = round(float(np.mean([out[k] for k in per if k in out])), 4)
    out["rho"] = round(float(np.mean(rho)), 3)
    return out


def greedy_ensemble(cp, folds, all_probs, k):
    """Caruana-style forward selection with replacement on OOF probs."""
    chosen, best = [], -1.0
    for _ in range(k):
        cand = []
        for j in range(len(all_probs)):
            trial = chosen + [j]
            avg = [np.mean([all_probs[i][f] for i in trial], axis=0)
                   for f in range(len(folds))]
            cand.append((score_probs(cp, folds, avg)["mean"], j))
        sc, j = max(cand)
        if sc <= best + 1e-9:
            break
        chosen.append(j)
        best = sc
    avg = [np.mean([all_probs[i][f] for i in chosen], axis=0) for f in range(len(folds))]
    return chosen, score_probs(cp, folds, avg)


# -----------------------------------------------------------------------------
# parsing
# -----------------------------------------------------------------------------
def _parse_value(v):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def parse_kv(s):
    return {k.strip(): _parse_value(v.strip())
            for k, v in (item.split("=") for item in s.split(",") if item)} if s else {}


def parse_grid(s):
    if not s:
        return [dict()]
    axes = {}
    for item in s.split(";"):
        k, vs = item.split("=")
        axes[k.strip()] = [_parse_value(v.strip()) for v in vs.split(",")]
    return [dict(zip(axes, combo)) for combo in itertools.product(*axes.values())]


_POOL_CTX = {}


def _worker(i):
    cp, grid, folds = _POOL_CTX["cp"], _POOL_CTX["grid"], _POOL_CTX["folds"]
    return i, evaluate(cp, grid[i], folds)


def _make_pool(n):
    """Process pool where fork exists (Linux/macOS: the cached covariances
    are shared copy-on-write); thread pool otherwise (Windows: numpy's
    linear algebra releases the GIL, so threads still overlap)."""
    import multiprocessing as mp
    if "fork" in mp.get_all_start_methods():
        return mp.get_context("fork").Pool(n), "processes"
    from multiprocessing.pool import ThreadPool
    return ThreadPool(n), "threads"


# -----------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("npz", nargs="+", help="first = primary (test) dataset")
    p.add_argument("--cv", choices=["forward", "loso", "both"], default="both")
    p.add_argument("--no-cv", action="store_true", help="fit the first config only")
    p.add_argument("--grid", default="", help='"bands=default,physio;windows=full,multi"')
    p.add_argument("--set", default="", help='fixed params "C=0.05,prior_weight=2"')
    p.add_argument("--ensemble", type=int, default=1,
                   help="max ensemble size picked greedily from the grid (1 = best single)")
    p.add_argument("-j", type=int, default=1, help="parallel grid workers")
    p.add_argument("--max-minutes", type=float, default=0,
                   help="stop scoring new grid configs after this many minutes (0 = no limit)")
    p.add_argument("--out", default="submission_fbts")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    print("data:")
    cp = Corpus(args.npz)
    base = parse_kv(args.set)
    grid = [dict(base, **g) for g in parse_grid(args.grid)]
    if len(args.npz) == 1:
        for g in grid:
            g.pop("data", None); g.pop("channels", None); g.pop("w_extra", None)
        grid = [dict(t) for t in {json.dumps(g, sort_keys=True): g for g in grid}.values()]

    # covariances up front (shared by all folds / configs)
    for g in grid:
        trainer, _, feat = split_cfg(g)
        cp.covs(trainer["channels"], feat)

    chosen = [0]
    cv_log = {}
    if not args.no_cv:
        folds = make_folds(cp, args.cv)
        if not folds:
            sys.exit("no CV folds possible on the primary dataset")
        print(f"CV ({args.cv}, {len(folds)} folds on {cp.names[0]}) over "
              f"{len(grid)} configuration(s)", flush=True)
        t0 = time.time()
        budget = args.max_minutes * 60 if args.max_minutes > 0 else np.inf
        results = [None] * len(grid)

        def done_line(i):
            el = (time.time() - t0) / 60
            n = sum(r is not None for r in results)
            print(f"  [{n}/{len(grid)}, {el:.1f} min] {results[i][0]}  <- {grid[i]}",
                  flush=True)

        if args.j > 1 and len(grid) > 1:
            _POOL_CTX.update(cp=cp, grid=grid, folds=folds)
            pool, kind = _make_pool(args.j)
            print(f"  {args.j} parallel {kind}")
            with pool:
                for i, res in pool.imap_unordered(_worker, range(len(grid))):
                    results[i] = res
                    done_line(i)
                    if time.time() - t0 > budget:
                        print("  time budget reached: stopping the grid")
                        pool.terminate()
                        break
        else:
            for i, g in enumerate(grid):
                if time.time() - t0 > budget:
                    print(f"  time budget reached: {len(grid) - i} config(s) skipped")
                    break
                if args.verbose:
                    print("  params:", g)
                results[i] = evaluate(cp, g, folds, verbose=args.verbose)
                done_line(i)
        keep = [i for i, r in enumerate(results) if r is not None]
        if not keep:
            sys.exit("no configuration finished inside the time budget")
        grid = [grid[i] for i in keep]
        results = [results[i] for i in keep]
        print(f"CV took {(time.time() - t0) / 60:.1f} min\nranking:")
        order = sorted(range(len(grid)), key=lambda i: -results[i][0]["mean"])
        for i in order:
            print(f"  {results[i][0]}  <- {grid[i]}")
        chosen = [order[0]]
        ens_score = None
        if args.ensemble > 1 and len(grid) > 1:
            chosen, ens_score = greedy_ensemble(cp, folds, [r[1] for r in results],
                                                args.ensemble)
            print(f"ensemble ({len(chosen)} picks): {ens_score}")
            for i in sorted(set(chosen)):
                print(f"   x{chosen.count(i)}  {grid[i]}")
            if len(set(chosen)) == 1:
                print("   (one configuration won — shipping it as a single model)")
        print(f"best single: {results[order[0]][0]}  <- {grid[order[0]]}")
        cv_log = dict(cv=args.cv, n_folds=len(folds),
                      ranking=[dict(score=results[i][0], config=grid[i]) for i in order],
                      ensemble=ens_score)

    # final fit on everything
    uniq = sorted(set(chosen))
    weights = [chosen.count(i) for i in uniq]
    members = []
    for i in uniq:
        t0 = time.time()
        m, _ = fit_member(cp, grid[i], cp.primary.copy())
        members.append(m)
        print(f"final fit {grid[i]}: {time.time() - t0:.0f}s", flush=True)
    model = members[0] if len(members) == 1 else FBTSEnsemble(members, weights)

    # inference latency on real primary windows through the full path
    model.set_input_montage(cp.ch[0])
    model.reset_state()
    Xp = cp.files[0]["X"][:40]
    t0 = time.time()
    for i in range(len(Xp)):
        model.predict(Xp[i:i + 1])
    ms = 1e3 * (time.time() - t0) / len(Xp)
    print(f"inference: {ms:.0f} ms/window -> ~{ms * SEALED_WINDOWS_GUESS / 6e4:.0f} min "
          f"for ~{SEALED_WINDOWS_GUESS} sealed windows on this CPU (limit 60 min)")
    model.reset_state()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / WEIGHTS_FILE)
    shutil.copyfile(Path(__file__).resolve().parent / "submission.py", out / "submission.py")
    (out / "training_info.json").write_text(json.dumps(dict(
        members=[grid[i] for i in uniq], weights=weights,
        files=[str(f) for f in args.npz], primary_channels=cp.ch[0],
        command=" ".join(sys.argv), python=platform.python_version(),
        platform=platform.platform(), inference_ms_per_window=round(ms, 1)), indent=2))
    if cv_log:
        (out / "cv_results.json").write_text(json.dumps(cv_log, indent=2))
    zpath = out.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in ("submission.py", WEIGHTS_FILE):
            zf.write(out / f, f)
    print(f"submission folder: {out}   zip: {zpath}")
    return out


if __name__ == "__main__":
    main()
