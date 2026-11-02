import numpy as np
import pandas as pd

from liqdiv.analysis import (
    constant_spread_book,
    correlation_greedy,
    inside_spread_curve,
    oracle_greedy,
    residuals,
    venues_to_target,
)
from liqdiv.data import to_grid


def test_to_grid_previous_tick_and_staleness():
    ts = np.array([10, 20, 20, 50])
    bid = np.array([1.0, 2.0, 2.5, 3.0])
    ask = bid + 0.1
    grid = np.array([5, 10, 19, 20, 49, 200])
    b, a = to_grid(ts, bid, ask, grid, max_stale_us=100)
    # before first quote -> nan; exact hit included; duplicate ts -> last row;
    # grid 200 sees the ts=50 quote, age 150 > 100 -> stale -> nan
    np.testing.assert_array_equal(b, [np.nan, 1.0, 1.0, 2.5, 2.5, np.nan])
    np.testing.assert_allclose(a[1:5], b[1:5] + 0.1)
    b2, _ = to_grid(ts, bid, ask, grid, max_stale_us=1000)
    assert b2[5] == 3.0


def test_inside_spread_constant_identity_and_monotone():
    rng = np.random.default_rng(0)
    R = rng.normal(size=(1000, 6))
    s = 2.0
    B, A = constant_spread_book(R, s)
    order = list(range(6))
    curve = inside_spread_curve(B, A, order)
    assert np.isclose(curve[0], s)
    assert np.all(np.diff(curve) <= 1e-12)
    expected_last = s - (R.max(axis=1) - R.min(axis=1)).mean()
    assert np.isclose(curve[-1], expected_last)


def test_correlation_greedy_picks_independent_venues_first():
    rng = np.random.default_rng(1)
    common = rng.normal(size=5000)
    # venues 0,1,2 share a common component; 3 and 4 are idiosyncratic
    R = np.column_stack(
        [common + 0.3 * rng.normal(size=5000) for _ in range(3)]
        + [rng.normal(size=5000) for _ in range(2)]
    )
    C = np.corrcoef(R.T)
    asc = correlation_greedy(C, ascending=True)
    desc = correlation_greedy(C, ascending=False)
    # first pick is an independent venue; after it, everything is ~uncorrelated
    # with the aggregator, so step 2 is a near-tie - but by step 3 the second
    # independent venue must be in (one common-factor venue is already present).
    assert asc[0] in {3, 4}
    assert {3, 4} <= set(asc[:3])
    assert set(desc[:3]) == {0, 1, 2}


def test_oracle_beats_or_ties_any_order():
    rng = np.random.default_rng(2)
    R = rng.normal(size=(2000, 5)) * np.array([0.1, 1, 1, 0.5, 2])
    B, A = constant_spread_book(R, 1.0)
    oracle = inside_spread_curve(B, A, oracle_greedy(B, A))
    for _ in range(20):
        other = inside_spread_curve(B, A, list(rng.permutation(5)))
        assert oracle[1] <= other[1] + 1e-12  # best pair is exhaustive


def test_venues_to_target():
    assert venues_to_target(np.array([1.0, 0.8, 0.7]), 0.75) == 3
    assert venues_to_target(np.array([1.0, 0.9]), 0.5) is None


def test_residuals_detrend_removes_constant_basis():
    idx = pd.date_range("2026-01-01", periods=600, freq="1s", tz="UTC")
    ref = np.full(600, 100.0)
    rng = np.random.default_rng(3)
    noise = rng.normal(scale=1e-5, size=600)
    bid = pd.DataFrame({"p": ref - 0.01, "x": ref * (1 + 6e-4 + noise) - 0.01}, index=idx)
    ask = bid + 0.02
    res = residuals(bid, ask, ["x"], reference="primary", primary="p", detrend_points=60)
    assert abs(res.r["x"].mean()) < 0.05  # 6 bp basis removed
    raw = residuals(bid, ask, ["x"], reference="primary", primary="p", detrend_points=0)
    assert 5.5 < raw.r["x"].mean() < 6.5
