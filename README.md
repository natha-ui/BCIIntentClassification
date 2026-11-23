# FB-TS: filter-bank Riemannian decoder with causal re-centring

A lightweight, calibration-free EEG decoder for **Track 02 (BCI decoding)** of the
[EEG/EMG Foundation Challenge 2026](https://github.com/neural-interfaces26/2026-competition).
It classifies 4-s windows into three cued mental commands: motor imagery, mental subtraction
and word generation. It is one subject-agnostic model with label-free, strictly causal
adaptation inside each session.

- **No deep learning, no GPU.** The pipeline is numpy, scipy and scikit-learn. Training takes
  minutes and inference ~20 ms per window on a laptop CPU.
- **Cross-subject and cross-session.** Each session is re-centred online by its running
  Riemannian mean, which removes most of the session and subject shift without labels.
- Runs on **Windows, macOS and Linux**. CI tests both Windows and Ubuntu.

> **Status.** The competition closed on 21 Nov 2026. This repo was not entered in the sealed
> phase, so it has no official score. The numbers below come from our own streamed
> cross-validation on the public PROTEUS release. The code still produces Codabench-format
> zips.

## Results (public PROTEUS, streamed CV, cell-balanced accuracy, chance = 0.333)

| model | later sessions of seen subjects (4 folds) | unseen subjects, LOSO (10 folds) | mean |
| --- | --- | --- | --- |
| FB-TS v1 (prior_weight 8, alpha_min 0) | 0.592 | 0.569 | 0.580 |
| **FB-TS v2** (prior_weight 2, alpha_min 0.03), the `v2` preset | **0.606** | **0.567** | **0.587** |

For reference, the organisers' public cross-subject baselines (2 test subjects, a different
split) were: constant 0.333, EEGNet 0.326, REVE probe 0.360. More detail, including the
adaptation grid, is in [docs/RESULTS.md](docs/RESULTS.md).

## Method

```
window (41 ch x 480 samples @ 120 Hz)
  -> filter bank (11 Butterworth bands, 4-55 Hz; zero-phase within the window)
  -> per band: OAS-shrunk covariance, trace-normalised
  -> whiten by the session's running Riemannian mean          <- causal, label-free
       M_t = M_{t-1}^1/2 (M_{t-1}^-1/2 C_t M_{t-1}^-1/2)^a_t M_{t-1}^1/2,
       a_t = max(1/(prior_weight + t), alpha_min), M_0 = pooled training mean
  -> tangent space at the training reference -> concatenate bands
  -> standardise -> multinomial logistic regression (class-balanced, C = 0.05)
```

- **Training:** each training session is re-centred by its own mean (`train_mode=batch`).
- **Test:** the state resets at every session start (`reset_state()`), and the stream runs on
  a `deepcopy`. Window *t* is whitened only by windows before it, so nothing crosses a session
  boundary and no test statistics are batch or transductive.
- **Ensembles:** `FBTSEnsemble` averages the probabilities of several members (band sets,
  time windows, channel subsets). Every member keeps its own adaptation state.

## Quick start

All commands are single lines, so they work in PowerShell, cmd, bash and zsh.

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py check
```

- If PowerShell blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
  once.
- Optional: `pip install torch --index-url https://download.pytorch.org/whl/cpu` makes `replay`
  feed torch tensors, exactly like the evaluation worker.

**Dry run, with no download (~5 min):**

```bash
python run_pipeline.py data --synthetic --out EMData/synth.npz
python run_pipeline.py all --preset v2-nocv --data EMData/synth.npz
```

**Real data.** PROTEUS is [NEMAR nm000290](https://nemar.org), a few GB:

```bash
python -m nemar download nm000290 -o data/nm000290
python run_pipeline.py data --bids data/nm000290
```

Expect **2655 windows (885/886/884), 10 subjects, 14 sessions, 41 channels**.

**Train, score and package:**

| preset | what it runs | approx. time |
| --- | --- | --- |
| `v2-nocv` | fit the v2 model and package it (no CV) | ~5 min |
| `v2` | v2 plus its streamed CV score (should reproduce the table above) | 20–35 min |
| `ablations` | recenter on/off × trace_norm on/off × prior_correction 0/0.5 | 45–90 min with `-j 4` |
| `features-small` | bands {default, physio, wide} × windows {full, trim}; greedy ensemble of ≤ 3 | 45–90 min with `-j 3` |
| `features` | bands {default, physio, fbcsp, wide} × windows {full, trim, multi}; ensemble of ≤ 3 | several hours |

```bash
python run_pipeline.py all --preset v2
python run_pipeline.py all --preset features-small -j 3 --minutes 90
```

Each run writes the following:
- `runs/<preset>/`: `submission.py`, `fbts_weights.npz`, `training_info.json`,
  `cv_results.json`, `environment.json`.
- `runs/<preset>.zip`: the upload format, with `submission.py` and the weights at the zip
  root.
- `runs/<preset>_run_log.txt`: everything that was printed.

Options:
- `-j N` scores N grid configurations in parallel, using processes on Linux/macOS and threads
  on Windows. Keep N ≤ RAM in GB / 3.
- `--minutes` caps the grid's time but keeps the configurations already scored.

**Score keys:**
- `forward`: later sessions of seen subjects.
- `loso`: unseen subjects.
- `mean`: the average of the two.
- `rho`: mean largest predicted-class fraction per session. About 0.33 is healthy; near 1
  means collapse.

Treat `forward` differences under 1–3 points as noise, because it rests on only 4 folds.

**Evaluator-style check of a package:**

```bash
python run_pipeline.py replay runs/v2 --sessions 0
```

This loads the folder's `submission.py` the way the worker does: `Solver().load_model(meta)`,
then a `deepcopy` plus `reset_state()` per session and B = 1 `predict`. It checks the output
type, shape and range, the zip layout, that nothing is written to the folder, and the latency.
The accuracy it prints is in-sample, so use it as a pipeline check only.

## Repository layout

| file | role |
| --- | --- |
| `submission.py` | model + `Solver(CompetSolver)`: the single file the evaluator runs. Inference needs only numpy + scipy (mne optional, for montage interpolation). |
| `train_fbts.py` | streamed CV (forward-session + LOSO), grid, greedy (Caruana) ensemble, final fit, packaging |
| `prepare_data.py` | datasets → `.npz` windows with the competition's preprocessing (50/60 Hz notch, 0.1–75 Hz band-pass, 120 Hz, 4 s from cue + 1 s, µV). Supports PROTEUS BIDS, plus Scherer2015, Dreyer2023, Stieger2021 and Zyma2019 as optional extra corpora. |
| `run_pipeline.py` | one entry point: `check`, `data`, `train`, `replay`, `all` |
| `make_synthetic.py` | contract-shaped synthetic data for tests and dry runs |
| `tests/` | unit tests: batch-equals-streaming, session isolation, save/load, solver contract |

## Competition contract and rules followed

- `predict(X)` receives one window at a time, `(1, 41, 480)`, with no metadata, and returns
  `(1,)` class indices.
- It does causal, label-free adaptation within a session only. Nothing crosses a session,
  and nothing is downloaded or written during evaluation.
- It trains only on public data (PROTEUS by default). Any extra corpus must be declared.

## Data

Datasets are **not** included and are never committed (see `.gitignore`). Each keeps its own
licence; check those before redistributing any derived file.

## References

- Barachant et al., "Multiclass brain–computer interface classification by Riemannian
  geometry", *IEEE TBME* 2012: tangent-space classification.
- Zanini et al., "Transfer learning: a Riemannian geometry framework with applications to
  brain–computer interfaces", *IEEE TBME* 2018: Riemannian re-centring.
- Chen et al., "Shrinkage algorithms for MMSE covariance estimation", *IEEE TSP* 2010: OAS.

## License

MIT; see [LICENSE](LICENSE).
