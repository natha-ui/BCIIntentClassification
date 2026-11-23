"""
make_synthetic.py — contract-shaped synthetic data (41 ch, 120 Hz, 480
samples, 3 classes, several subjects x sessions x 2 contexts) in the
prepare_data.py npz format, for end-to-end checks of train_fbts.py and
submission.py without any download.

Deliberately hard: weak class effects, 1/f background sources, per-session
cap re-placement / gain / brain-state drift, and per-subject mixing so that
the unseen-subject (loso) condition is meaningfully different from the
cross-session (forward) one.

    python make_synthetic.py --out data/synth.npz --subjects 4 --sessions 3
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.signal import butter, sosfiltfilt

from prepare_data import save_npz

FS, N_CH, N_T = 120.0, 41, 480
CH_NAMES = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2",
            "FC6", "T7", "C3", "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6",
            "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2", "AF3", "AF4",
            "F5", "F1", "F2", "F6", "FC3", "FC4", "C5", "C1", "C2", "C6", "CP3"]
N_SRC, N_BG = 12, 30


def simulate(rng, n_subjects, n_sessions, per_class_ctx=15):
    bands = [(4, 8), (8, 12), (18, 26), (30, 45)]
    sos = [butter(4, b, "bandpass", fs=FS, output="sos") for b in bands]
    freqs = np.fft.rfftfreq(N_T, 1 / FS)
    pink = 1.0 / np.sqrt(np.maximum(freqs, 1.0))
    # class effect shared across subjects (what transfers), plus a
    # subject-specific part (what does not)
    effect_common = np.where(rng.random((3, len(bands), N_SRC)) < 0.3,
                             rng.choice([0.6, 1.6], size=(3, len(bands), N_SRC)), 1.0)
    X, y, subj, sess, ctx = [], [], [], [], []
    A0 = rng.standard_normal((N_CH, N_SRC))
    for s in range(n_subjects):
        A = A0 + 0.5 * rng.standard_normal((N_CH, N_SRC))
        A_bg = rng.standard_normal((N_CH, N_BG))
        effect = effect_common * np.where(
            rng.random((3, len(bands), N_SRC)) < 0.15,
            rng.choice([0.7, 1.4], size=(3, len(bands), N_SRC)), 1.0)
        for t in range(1, n_sessions + 1):
            drift = np.eye(N_CH) + 0.6 * rng.standard_normal((N_CH, N_CH)) / np.sqrt(N_CH)
            gains = np.exp(0.4 * rng.standard_normal(N_CH)) * np.exp(0.5 * rng.standard_normal())
            state = np.exp(0.4 * rng.standard_normal(len(bands)))
            A_s, Abg_s = gains[:, None] * (drift @ A), gains[:, None] * (drift @ A_bg)
            for c_name in ("graz", "brainhero"):
                labels = np.repeat(np.arange(3), per_class_ctx)
                rng.shuffle(labels)
                for c in labels:
                    src = np.zeros((N_SRC, N_T))
                    for b, sb in enumerate(sos):
                        src += state[b] * effect[c, b][:, None] * \
                            sosfiltfilt(sb, rng.standard_normal((N_SRC, N_T)))
                    spec = (rng.standard_normal((N_BG, len(freqs))) +
                            1j * rng.standard_normal((N_BG, len(freqs)))) * pink
                    bg = np.fft.irfft(spec, n=N_T) * np.exp(0.3 * rng.standard_normal((N_BG, 1)))
                    bg *= 1.0 / bg.std()
                    x = A_s @ src + Abg_s @ bg + 0.5 * rng.standard_normal((N_CH, N_T))
                    X.append(10 * x)           # ~µV scale
                    y.append(c); subj.append(f"synth/sub-{s:02d}")
                    sess.append(f"ses-{t:02d}"); ctx.append(c_name)
    return np.stack(X).astype(np.float32), np.array(y), subj, sess, ctx


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synth.npz")
    p.add_argument("--subjects", type=int, default=4)
    p.add_argument("--sessions", type=int, default=3)
    p.add_argument("--per-class-ctx", type=int, default=15)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    X, y, subj, sess, ctx = simulate(rng, args.subjects, args.sessions,
                                     args.per_class_ctx)
    save_npz(args.out, X, y, subj, sess, ctx, CH_NAMES, "synthetic")


if __name__ == "__main__":
    main()
