# Results log (public PROTEUS release, NEMAR nm000290 v1.0.0)

## Evaluation protocol

- Every test session is streamed like the official evaluator: `reset_state()`, then windows
  in recording order.
- The metric is balanced accuracy per subject × session × interface (Graz / BrainHero),
  averaged over cells.
- **forward:** for each subject with ≥ 2 sessions, train on all other subjects plus that
  subject's earlier sessions, then test each later session. There are 4 folds: P04 and P10,
  sessions 2–3.
- **loso:** leave one subject out (10 folds).
- The data has 2655 windows (885 MI / 886 subtraction / 884 word), 10 subjects, 14 sessions
  and 41 channels.

## 2026-10-23: first real-data run (defaults: prior_weight 8, batch training, alpha_min 0)

| C | forward | loso | mean |
| --- | --- | --- | --- |
| 0.05 | 0.592 | 0.569 | 0.580 |
| 0.2 | 0.590 | 0.569 | 0.580 |

C has almost no effect.

## 2026-10-24: adaptation grid (C = 0.05, 12 configs)

- **Best:** prior_weight 2, batch, alpha_min 0.03, scoring **0.606 / 0.567 / 0.587**. This is
  the `v2` preset.
- prior_weight 32 is clearly the worst (mean 0.546–0.570); 2 ≈ 8. Fast adaptation to the
  session matters.
- alpha_min 0.03 ≥ 0 in 6 of 6 pairs (+0.3 to +2.0 points). It pays to keep tracking drift
  within a session.
- Online training mode raises forward (up to 0.619) but lowers LOSO by 1–2 points: a tie
  overall.
- **Conclusion:** the adaptation knobs are saturated, so further gains must come from the
  features, the classifier or more data.

## Synthetic checks

The synthetic data is contract-shaped, with session drift and subject mixing.

| re-centring | forward | loso |
| --- | --- | --- |
| on | 0.99 | 0.96 |
| off | 0.84 | 0.71 |

## Not yet run on real data

- Feature grid (bands × windows, ensembles).
- Pooling with Scherer2015 / Dreyer2023.
- Ablations: recenter off, trace_norm off, prior_correction 0.5.

`prior_correction` had no effect inside CV before 2026-11-23. CV scored whole sessions at
once, and the running class prior was updated only after the batch. This is now fixed
(window by window), and `tests/` checks that batch output matches streaming.

## Reference: organisers' public baselines

These used a cross-subject split with 2 test subjects and seeds 33–35, so the splits differ
from ours.

| model | balanced accuracy |
| --- | --- |
| Constant | 0.333 |
| EEGNet | 0.326 |
| REVE probe | 0.360 |
