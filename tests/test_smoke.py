"""Fast checks of the contract-critical behaviour (run: python -m pytest -q)."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from make_synthetic import simulate  # noqa: E402
from submission import FBTSEnsemble, FBTSModel, Solver, load_any  # noqa: E402


@pytest.fixture(scope="module")
def data():
    X, y, subj, sess, _ = simulate(np.random.default_rng(0), 2, 2, per_class_ctx=6)
    groups = np.array([f"{a}/{b}" for a, b in zip(subj, sess)])
    return X, y, groups


@pytest.fixture(scope="module")
def model(data):
    X, y, groups = data
    return FBTSModel(bands="wide", prior_weight=2, alpha_min=0.03).fit(X, y, groups)


def _stream(m, X):
    m = copy.deepcopy(m)
    m.reset_state()
    return np.concatenate([m.predict_proba(X[i:i + 1]) for i in range(len(X))])


@pytest.mark.parametrize("pc", [0.0, 0.5])
def test_batch_equals_streaming(model, data, pc):
    """CV scores whole sessions at once; the evaluator sends B=1 windows.
    Both must give identical probabilities (incl. the prior correction)."""
    X = data[0][:30]
    m = copy.deepcopy(model)
    m.prior_correction = pc
    m.reset_state()
    np.testing.assert_allclose(m.predict_proba(X), _stream(m, X), atol=1e-10)


def test_reset_state_isolates_sessions(model, data):
    X = data[0][:20]
    m = copy.deepcopy(model)
    m.reset_state(); a = m.predict_proba(X)
    m.predict_proba(data[0][20:40])          # adapt to something else
    m.reset_state(); b = m.predict_proba(X)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_save_load_roundtrip(model, data, tmp_path):
    X = data[0][:10]
    for obj in (model, FBTSEnsemble([model, model], [1, 2])):
        obj.save(tmp_path / "w.npz")
        np.testing.assert_allclose(_stream(obj, X), _stream(load_any(tmp_path / "w.npz"), X),
                                   atol=1e-12)


def test_solver_contract(model, data, tmp_path):
    model.save(tmp_path / "fbts_weights.npz")
    meta = dict(submission_dir=tmp_path, sfreq=120.0, ch_names=None, device="cpu")
    m = Solver().load_model(meta)
    m = copy.deepcopy(m)
    m.reset_state()
    out = m.predict(data[0][:1].astype(np.float32))
    assert out.shape == (1,) and np.issubdtype(out.dtype, np.integer) and 0 <= out[0] <= 2
