"""
submission.py — Filter-Bank Riemannian Tangent-Space decoder (FB-TS) with
causal, label-free Riemannian re-centring, for Track 02 (BCI decoding,
cross-session) of the EEG/EMG Foundation Challenge 2026.

Contract (benchmark_utils/base_solver.py, scientific_faq_bci.md, 2026-10):
  * predict(X) receives ONE window at a time: float32 (1, 41, 480) — 41 EEG
    channels, 4 s starting 1 s after cue onset, 0.1-75 Hz band-passed,
    50/60 Hz notched, resampled to 120 Hz, in microvolts.  No metadata.
  * Return one class index per window, shape (B,): 0 = kinesthetic motor
    imagery, 1 = mental subtraction, 2 = word generation.
  * One stream = one participant x session, windows chronological.  Each
    stream runs on copy.deepcopy(model) after reset_state().  Causal,
    label-free adaptation inside a stream is allowed; nothing may cross the
    session boundary (this is what the running re-centring mean does).
  * Single file; only what the worker image carries.  Inference here needs
    numpy + scipy only (mne optional, for montage interpolation).

Pipeline (one "member")
-----------------------
  window (C, T) -> [montage map / channel subset to the member's montage]
  -> filter bank (K zero-phase Butterworth bands) -> crop W time windows
  -> per (band, window) block: OAS-shrunk covariance, trace-normalised
  -> whiten by the session's running Riemannian mean (causal)
  -> tangent space at the training reference -> concatenate
  -> standardise -> multinomial logistic regression (numpy at inference)

An FBTSEnsemble averages the class probabilities of several members (e.g.
different band sets / windows / channel subsets / training corpora); every
member keeps its own causal re-centring state.

Weights file: fbts_weights.npz (a single member or an ensemble; files from
the first version load unchanged).  Training: train_fbts.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

try:  # the competition harness; absent when the file is used standalone
    from benchmark_utils.base_solver import CompetSolver
except Exception:  # pragma: no cover
    class CompetSolver:  # minimal stand-in so this module imports anywhere
        name = "FBTS"

        def load_model(self, meta):
            raise NotImplementedError

        def fit(self, model, train_loader):
            pass

        def save_model(self, model, path):
            pass


WEIGHTS_FILE = "fbts_weights.npz"
TARGET_SFREQ = 120.0

# Filter-bank presets (Hz).  Input is 0.1-75 Hz band-passed at 120 Hz
# (Nyquist 60), so top edges stay <= 55 Hz.  Each band x window block adds
# C(C+1)/2 features (861 for 41 channels).
BAND_SETS = {
    # theta, overlapping mu/beta, low gamma, broadband (first version)
    "default": ((4, 8), (8, 12), (10, 14), (12, 16), (16, 20), (20, 24),
                (24, 28), (28, 32), (32, 40), (40, 55), (4, 40)),
    # classic FBCSP 4-Hz bank
    "fbcsp": ((4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28),
              (28, 32), (32, 36), (36, 40)),
    # the referenced fatbaby614 FBTS repo (mu / beta only)
    "repo": ((8, 10), (10, 16), (16, 24), (24, 32)),
    # physiological bands incl. delta/theta (mental tasks are not only mu/beta)
    "physio": ((1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45)),
    # few wide bands: cheap, low-variance
    "wide": ((4, 8), (8, 13), (13, 30), (30, 45), (4, 45)),
    # 2-Hz resolution in the mu/beta range + theta and gamma
    "fine": ((4, 8), (8, 10), (10, 12), (12, 14), (14, 16), (16, 18),
             (18, 20), (20, 24), (24, 28), (28, 32), (32, 45)),
}

# Time-window presets, seconds inside the 4-s input (= 1-5 s after the cue;
# PROTEUS feedback starts 3 s after the cue = 2 s into the input).
WINDOW_SETS = {
    "full": ((0.0, 4.0),),
    "trim": ((0.5, 4.0),),
    "early": ((0.0, 2.5),),
    "late": ((1.5, 4.0),),
    "halves": ((0.0, 2.0), (2.0, 4.0)),
    "multi": ((0.0, 4.0), (0.0, 2.0), (2.0, 4.0)),
}

# old 10-20 names -> 10-10 names, so montages from other corpora match
CH_ALIASES = {"t3": "t7", "t4": "t8", "t5": "p7", "t6": "p8"}


def _norm(name):
    n = str(name).strip().lower()
    return CH_ALIASES.get(n, n)


def resolve_bands(bands):
    return BAND_SETS[bands] if isinstance(bands, str) else bands


def resolve_windows(windows):
    return WINDOW_SETS[windows] if isinstance(windows, str) else windows


# =============================================================================
# 1. SPD-matrix primitives (batched over leading axes, eigh-based)
# =============================================================================
def _eig_fn(C, fn):
    """U diag(fn(w)) U^T for (a batch of) symmetric matrices C (..., n, n)."""
    w, U = np.linalg.eigh(C)
    return (U * fn(w)[..., None, :]) @ np.swapaxes(U, -1, -2)


def sqrtm(C):    return _eig_fn(C, np.sqrt)
def invsqrtm(C): return _eig_fn(C, lambda w: 1.0 / np.sqrt(w))
def logm(C):     return _eig_fn(C, np.log)
def expm_sym(C): return _eig_fn(C, np.exp)
def powm(C, p):  return _eig_fn(C, lambda w: w ** p)


def sqrt_and_invsqrt(C):
    """(C^1/2, C^-1/2) from a single eigendecomposition."""
    w, U = np.linalg.eigh(C)
    Ut = np.swapaxes(U, -1, -2)
    s = np.sqrt(w)
    return (U * s[..., None, :]) @ Ut, (U * (1.0 / s)[..., None, :]) @ Ut


def karcher_mean(C, tol=1e-5, max_iter=50):
    """Affine-invariant Riemannian mean of SPD matrices C (n_mat, n, n).

    Riemannian gradient descent with pyRiemann's adaptive step rule,
    initialised at the log-Euclidean mean.  Deterministic.
    """
    C = np.asarray(C, dtype=np.float64)
    M = expm_sym(logm(C).mean(0))
    nu, tau = 1.0, np.inf
    for _ in range(max_iter):
        Ms, Mis = sqrt_and_invsqrt(M)
        T = logm(Mis @ C @ Mis).mean(0)
        crit = np.linalg.norm(T)
        if crit < tol:
            break
        M = Ms @ expm_sym(nu * T) @ Ms
        h = nu * crit
        if h < tau:
            nu, tau = 0.95 * nu, h
        else:
            nu *= 0.5
    return M


def tangent_vectors(C, ref_isqrt):
    """log(ref^-1/2 C ref^-1/2), upper triangle, off-diagonals x sqrt(2).

    C: (n_mat, n, n), ref_isqrt = ref^-1/2 -> (n_mat, n(n+1)/2).
    """
    n = C.shape[-1]
    S = logm(ref_isqrt @ C @ ref_isqrt)
    iu = np.triu_indices(n)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return S[:, iu[0], iu[1]] * coef


def oas_covariances(X):
    """Oracle-Approximating-Shrinkage covariances of windows X (n, p, t)
    (vectorised re-implementation of sklearn's OAS, per-channel demeaned)."""
    X = X - X.mean(axis=-1, keepdims=True)
    n_tr, p, n = X.shape
    S = (X @ np.swapaxes(X, -1, -2)) / n
    mu = np.trace(S, axis1=1, axis2=2) / p
    alpha = np.mean(S ** 2, axis=(1, 2))
    num = alpha + mu ** 2
    den = (n + 1.0) * (alpha - mu ** 2 / p)
    shrink = np.clip(np.where(den > 0, num / den, 1.0), 0.0, 1.0)
    return ((1 - shrink)[:, None, None] * S
            + (shrink * mu)[:, None, None] * np.eye(p))


# =============================================================================
# 2. Montage mapping
# =============================================================================
def montage_matrix(src_names, dst_names, k=4):
    """(n_dst, n_src) linear map from channels `src_names` to `dst_names`.

    Exact name matches (case-insensitive, T3/T4/T5/T6 aliased to T7/T8/P7/P8)
    are copied — a member trained on a channel SUBSET therefore just selects
    its channels.  Missing ones are inverse-distance interpolated from the k
    nearest available channels (MNE 10-05 positions); without positions they
    fall back to the mean of all src channels.
    """
    src_l = [_norm(s) for s in src_names]
    dst_l = [_norm(s) for s in dst_names]
    A = np.zeros((len(dst_l), len(src_l)))
    missing = [i for i, name in enumerate(dst_l) if name not in src_l]
    for i, name in enumerate(dst_l):
        if name in src_l:
            A[i, src_l.index(name)] = 1.0
    if not missing:
        return A
    pos = {}
    try:
        import warnings
        import mne
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                mont = mne.channels.make_standard_montage("colin27_1005")
            except Exception:
                mont = mne.channels.make_standard_montage("standard_1005")
        pos = {n.lower(): p for n, p in mont.get_positions()["ch_pos"].items()}
    except Exception:
        pass
    src_pos = np.array([pos.get(n, [np.nan] * 3) for n in src_l])
    ok = ~np.isnan(src_pos).any(1)
    for i in missing:
        p = pos.get(dst_l[i])
        if p is None or ok.sum() == 0:
            A[i] = 1.0 / len(src_l)
            continue
        d = np.linalg.norm(src_pos[ok] - np.asarray(p), axis=1)
        order = np.argsort(d)[:k]
        idx = np.flatnonzero(ok)[order]
        w = 1.0 / np.maximum(d[order], 1e-3)
        A[i, idx] = w / w.sum()
    return A


def common_channels(name_lists):
    """Channels present (by normalised name) in every list, in the order of
    the first list."""
    sets = [{_norm(n) for n in names} for names in name_lists[1:]]
    return [n for n in name_lists[0] if all(_norm(n) in s for s in sets)]


# =============================================================================
# 3. One member
# =============================================================================
class FBTSModel:
    """Filter-bank tangent-space decoder with causal Riemannian re-centring.

    Parameters
    ----------
    bands : preset name in BAND_SETS or list of (lo, hi) Hz.
    windows : preset name in WINDOW_SETS or list of (t0, t1) s in the window.
    sfreq : sampling rate the model expects (contract: 120 Hz).
    order : Butterworth order.
    trace_norm : divide every covariance by its trace (removes gain drift).
    recenter : whiten each window by the running session mean (causal).
    prior_weight : the training prior counts as this many windows in the
        running mean (small -> adapts fast).
    alpha_min : floor on the running-mean step (0 = cumulative mean;
        e.g. 0.03 keeps tracking slow within-session drift).
    train_mode : "batch" (re-centre each training session by its own mean)
        or "online" (replay the causal re-centring on training sessions).
    C : inverse L2 strength of the logistic regression.
    prior_correction : >0 enables an online class-prior (logit-bias)
        correction.  Off by default.
    """

    def __init__(self, bands="default", windows="full", sfreq=TARGET_SFREQ,
                 order=4, trace_norm=True, recenter=True, prior_weight=8.0,
                 alpha_min=0.0, train_mode="batch", C=0.05,
                 prior_correction=0.0, random_state=0):
        self.bands = [tuple(map(float, b)) for b in resolve_bands(bands)]
        self.windows = [tuple(map(float, w)) for w in resolve_windows(windows)]
        self.sfreq = float(sfreq)
        self.order = int(order)
        self.trace_norm = bool(trace_norm)
        self.recenter = bool(recenter)
        self.prior_weight = float(prior_weight)
        self.alpha_min = float(alpha_min)
        self.train_mode = train_mode
        self.C = float(C)
        self.prior_correction = float(prior_correction)
        self.random_state = int(random_state)
        self._build_filters()
        self.ch_names_ = None      # montage the weights were trained on
        self.classes_ = None
        self.ref_ = None           # (B, n, n) tangent-space reference
        self.prior_ = None         # (B, n, n) start of the running mean
        self.scale_mean_ = self.scale_std_ = None
        self.W_ = self.b_ = None
        self._A = None             # input -> member montage map
        self._ref_isqrt = None
        self.reset_state()

    # ---- construction helpers ------------------------------------------
    def _build_filters(self):
        # Bands are defined for 120 Hz; at a lower rate (e.g. the 100 Hz
        # Simulated smoke test) clip them below 0.9 x Nyquist.
        top = 0.9 * self.sfreq / 2
        self.bands = [(lo, min(hi, top)) for lo, hi in self.bands if lo < top - 1]
        self._sos = [butter(self.order, b, btype="bandpass", fs=self.sfreq,
                            output="sos") for b in self.bands]

    @property
    def n_blocks(self):
        return len(self.bands) * len(self.windows)

    def get_params(self):
        return dict(bands=self.bands, windows=self.windows, sfreq=self.sfreq,
                    order=self.order, trace_norm=self.trace_norm,
                    recenter=self.recenter, prior_weight=self.prior_weight,
                    alpha_min=self.alpha_min, train_mode=self.train_mode,
                    C=self.C, prior_correction=self.prior_correction,
                    random_state=self.random_state)

    @property
    def is_fitted(self):
        return self.W_ is not None

    def set_input_montage(self, ch_names):
        """Tell the model which channels arrive in predict (from meta)."""
        if ch_names is None or self.ch_names_ is None:
            self._A = None
        elif [_norm(c) for c in ch_names] == [_norm(c) for c in self.ch_names_]:
            self._A = None
        else:
            self._A = montage_matrix(ch_names, self.ch_names_)

    # ---- state -----------------------------------------------------------
    def reset_state(self):
        """Called by the evaluator at the start of every session stream."""
        self.M_ = None if self.prior_ is None else self.prior_.copy()
        self.n_seen_ = 0
        self.p_run_ = None
        self.n_pc_ = 0             # windows seen by the prior correction

    # ---- features --------------------------------------------------------
    def covariances(self, X, dtype=np.float64):
        """(n, C, T) windows -> (B, n, C, C) block covariances, B = bands x
        windows (band-major)."""
        X = np.asarray(X, dtype=np.float64)
        if self._A is not None:
            X = np.einsum("dc,nct->ndt", self._A, X)
        X = X - X.mean(axis=-1, keepdims=True)
        n, c, T = X.shape
        cuts = [(int(round(t0 * self.sfreq)), min(T, int(round(t1 * self.sfreq))))
                for t0, t1 in self.windows]
        out = np.empty((self.n_blocks, n, c, c), dtype=dtype)
        b = 0
        for sos in self._sos:
            padlen = min(3 * (2 * len(sos) + 1), T - 1)
            Xf = sosfiltfilt(sos, X, axis=-1, padlen=padlen)
            for s0, s1 in cuts:
                C = oas_covariances(Xf[..., s0:s1])
                if self.trace_norm:
                    C = C / np.trace(C, axis1=1, axis2=2)[:, None, None] * c
                out[b] = C
                b += 1
        return out

    def _features(self, covs):
        """(B, n, C, C) re-centred covariances -> (n, B*C(C+1)/2)."""
        if self._ref_isqrt is None:
            self._ref_isqrt = invsqrtm(self.ref_)
        return np.concatenate(
            [tangent_vectors(np.asarray(covs[k], np.float64), self._ref_isqrt[k])
             for k in range(covs.shape[0])], axis=1)

    def _logits(self, F):
        return ((F - self.scale_mean_) / self.scale_std_) @ self.W_ + self.b_

    # ---- causal re-centring ----------------------------------------------
    def recenter_stream(self, covs, prior=None, update=True):
        """Whiten window t by the running mean of windows < t (+ prior).

        covs: (B, n, C, C) of ONE stream in time order.  Uses and updates
        the per-stream state self.M_ / self.n_seen_.
        """
        B, n = covs.shape[:2]
        if self.M_ is None:
            self.M_ = (prior if prior is not None else self.prior_).copy()
        out = np.empty(covs.shape, dtype=np.float64)
        for i in range(n):
            Ms, Mis = sqrt_and_invsqrt(self.M_)               # (B, C, C)
            out[:, i] = Mis @ covs[:, i] @ Mis
            if update:
                self.n_seen_ += 1
                alpha = max(1.0 / (self.prior_weight + self.n_seen_),
                            self.alpha_min)
                # geodesic step M <- M^1/2 (M^-1/2 C M^-1/2)^alpha M^1/2
                self.M_ = Ms @ powm(out[:, i], alpha) @ Ms
        return out

    # ---- training --------------------------------------------------------
    def fit(self, X, y, groups, ch_names=None, covs=None, sample_weight=None):
        """X: (n, C, T) windows at self.sfreq (or None with `covs`);
        y: (n,) int labels; groups: (n,) session ids (subject x session),
        each group's windows in time order; covs: optional precomputed
        self.covariances(X); sample_weight: optional (n,) weights."""
        from sklearn.linear_model import LogisticRegression
        if self.train_mode not in ("batch", "online"):
            raise ValueError(f"unknown train_mode {self.train_mode!r}")
        y, groups = np.asarray(y), np.asarray(groups)
        self.ch_names_ = None if ch_names is None else list(ch_names)
        self._A = None
        self._ref_isqrt = None
        if covs is None:
            covs = self.covariances(X)
        B = covs.shape[0]
        # prior of the running mean: pooled Riemannian mean of all windows
        self.prior_ = np.stack([karcher_mean(covs[k]) for k in range(B)])
        if self.recenter:
            rc = np.empty(covs.shape, dtype=np.float32)
            for g in np.unique(groups):
                idx = np.flatnonzero(groups == g)
                if self.train_mode == "batch":
                    for k in range(B):
                        W = invsqrtm(karcher_mean(covs[k, idx]))
                        rc[k, idx] = W @ covs[k, idx] @ W
                else:
                    self.reset_state()
                    rc[:, idx] = self.recenter_stream(covs[:, idx])
            covs = rc
        lab = y >= 0
        covs = covs[:, lab]
        y = y[lab]
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)[lab]
        self.ref_ = np.stack([karcher_mean(covs[k]) for k in range(B)])
        self._ref_isqrt = None
        F = self._features(covs)
        self.scale_mean_ = F.mean(0)
        self.scale_std_ = F.std(0) + 1e-12
        clf = LogisticRegression(C=self.C, max_iter=3000,
                                 class_weight="balanced",
                                 random_state=self.random_state)
        clf.fit((F - self.scale_mean_) / self.scale_std_, y,
                sample_weight=sample_weight)
        self.classes_ = clf.classes_.astype(np.int64)
        self.W_ = clf.coef_.T.copy()
        self.b_ = clf.intercept_.copy()
        if self.W_.shape[1] == 1:  # sklearn binary convention -> 2 columns
            self.W_ = np.hstack([-self.W_, self.W_]) / 2
            self.b_ = np.array([-self.b_[0], self.b_[0]]) / 2
        self.reset_state()
        return self

    # ---- inference (the contract) ------------------------------------------
    def predict_proba(self, X):
        """X: (B, C, T) windows of ONE stream in time order (torch or numpy).
        Returns (B, n_classes) probabilities."""
        if not self.is_fitted:
            raise RuntimeError("FBTSModel is not trained: no weights found.")
        if hasattr(X, "detach"):
            X = X.detach().cpu().numpy()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2:
            X = X[None]
        return self.predict_proba_covs(self.covariances(X))

    def predict_proba_covs(self, covs):
        """predict_proba from precomputed (B, n, C, C) covariances of ONE
        stream in time order (used by the CV tooling)."""
        if self.recenter:
            covs = self.recenter_stream(covs)
        logits = self._logits(self._features(covs))
        if self.prior_correction <= 0:
            logits -= logits.max(1, keepdims=True)
            P = np.exp(logits)
            return P / P.sum(1, keepdims=True)
        # online class-prior correction, strictly window by window, so a
        # whole-stream call (CV) gives the same result as B=1 streaming
        if self.p_run_ is None:
            self.p_run_ = np.full(logits.shape[1], 1.0 / logits.shape[1])
        P = np.empty_like(logits)
        for i, z in enumerate(logits):
            z = z - self.prior_correction * np.log(self.p_run_ + 1e-6)
            p = np.exp(z - z.max())
            P[i] = p / p.sum()
            self.n_pc_ += 1
            a = max(1.0 / (self.prior_weight + self.n_pc_), 0.02)
            self.p_run_ = (1 - a) * self.p_run_ + a * P[i]
        return P

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    # ---- persistence (plain npz, no pickle) --------------------------------
    def _state(self, prefix=""):
        return {prefix + "params": json.dumps(self.get_params()),
                prefix + "ch_names": json.dumps(self.ch_names_),
                prefix + "classes": self.classes_, prefix + "ref": self.ref_,
                prefix + "prior": self.prior_,
                prefix + "scale_mean": self.scale_mean_,
                prefix + "scale_std": self.scale_std_,
                prefix + "W": self.W_, prefix + "b": self.b_}

    @classmethod
    def _from_state(cls, z, prefix=""):
        params = json.loads(str(z[prefix + "params"]))
        m = cls(**params)
        m.ch_names_ = json.loads(str(z[prefix + "ch_names"]))
        m.classes_, m.ref_ = z[prefix + "classes"], z[prefix + "ref"]
        m.prior_ = z[prefix + "prior"]
        m.scale_mean_, m.scale_std_ = z[prefix + "scale_mean"], z[prefix + "scale_std"]
        m.W_, m.b_ = z[prefix + "W"], z[prefix + "b"]
        m.reset_state()
        return m

    def save(self, path):
        np.savez(Path(path), **self._state())

    @classmethod
    def load(cls, path):
        return cls._from_state(np.load(Path(path), allow_pickle=False))


# =============================================================================
# 4. Ensemble of members
# =============================================================================
class FBTSEnsemble:
    """Weighted average of member probabilities.  Each member applies its own
    montage map and causal re-centring, so members may differ in bands,
    windows, channel subsets and training corpora."""

    def __init__(self, members, weights=None):
        import copy
        members = list(members)
        # the same object twice would share (and double-update) one
        # re-centring state: give every slot its own copy
        self.members = [copy.deepcopy(m) if any(m is o for o in members[:i]) else m
                        for i, m in enumerate(members)]
        w = np.ones(len(self.members)) if weights is None else np.asarray(weights, float)
        self.weights = w / w.sum()
        cls0 = self.members[0].classes_
        if any(not np.array_equal(m.classes_, cls0) for m in self.members):
            raise ValueError("ensemble members disagree on classes")
        self.classes_ = cls0
        self.sfreq = self.members[0].sfreq

    def set_input_montage(self, ch_names):
        for m in self.members:
            m.set_input_montage(ch_names)

    def reset_state(self):
        for m in self.members:
            m.reset_state()

    def predict_proba(self, X):
        if hasattr(X, "detach"):
            X = X.detach().cpu().numpy()
        X = np.asarray(X, dtype=np.float64)
        return sum(w * m.predict_proba(X) for w, m in zip(self.weights, self.members))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    def save(self, path):
        state = {"n_members": np.int64(len(self.members)), "weights": self.weights}
        for i, m in enumerate(self.members):
            state.update(m._state(prefix=f"m{i}_"))
        np.savez(Path(path), **state)

    @classmethod
    def load(cls, path):
        z = np.load(Path(path), allow_pickle=False)
        n = int(z["n_members"])
        return cls([FBTSModel._from_state(z, f"m{i}_") for i in range(n)],
                   z["weights"])


def load_any(path):
    """Load a single-member or ensemble weights file."""
    z = np.load(Path(path), allow_pickle=False)
    if "n_members" in z.files:
        return FBTSEnsemble.load(path)
    return FBTSModel.load(path)


# =============================================================================
# 5. The benchopt / Codabench solver
# =============================================================================
def _to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _group_ids(info, n):
    """(subject, session) id per window of a training batch."""
    subj = _to_np(info.get("subject_id", np.zeros(n))).reshape(-1)
    sess = _to_np(info.get("stream_id", info.get("record_id", np.zeros(n)))
                  ).reshape(-1)
    return [f"{s}/{t}" for s, t in zip(subj, sess)]


class Solver(CompetSolver):
    """FB-TS + causal Riemannian re-centring.  Hyper-parameters are benchopt
    parameters (single member; ensembles are built with train_fbts.py):
        benchopt run tracks/bci_decoding -s "FBTS-Riemann[bands=[default,physio]]"
    """

    name = "FBTS-Riemann"

    parameters = {
        "C": [0.05],
        "prior_weight": [2.0],
        "alpha_min": [0.03],
        "trace_norm": [True],
        "recenter": [True],
        "train_mode": ["batch"],
        "prior_correction": [0.0],
        "bands": ["default"],
        "windows": ["full"],
    }

    def load_model(self, meta):
        weights = Path(meta["submission_dir"]) / WEIGHTS_FILE
        if weights.exists():
            model = load_any(weights)
        else:  # untrained: fit() must run (training=True) before predict
            model = FBTSModel(bands=self.bands, windows=self.windows,
                              sfreq=meta.get("sfreq", TARGET_SFREQ),
                              C=self.C, prior_weight=self.prior_weight,
                              alpha_min=self.alpha_min,
                              trace_norm=self.trace_norm,
                              recenter=self.recenter,
                              train_mode=self.train_mode,
                              prior_correction=self.prior_correction)
        if abs(float(meta.get("sfreq", model.sfreq)) - model.sfreq) > 1e-6:
            raise ValueError(f"model expects {model.sfreq} Hz, data is "
                             f"{meta['sfreq']} Hz")
        model.set_input_montage(meta.get("ch_names"))
        return model

    def fit(self, model, train_loader):
        X, y, groups = [], [], []
        for Xb, yb, info in train_loader:
            X.append(_to_np(Xb).astype(np.float32))
            y.append(_to_np(yb))
            groups += _group_ids(info, len(Xb))
        X, y = np.concatenate(X), np.concatenate(y)
        model.fit(X, y, np.asarray(groups), ch_names=self.meta.get("ch_names"))

    def save_model(self, model, path):
        model.save(Path(path) / WEIGHTS_FILE)
