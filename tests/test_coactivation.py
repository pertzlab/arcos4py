"""Tests for co-activation event detection (detect_coactivation_events and friends)."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.sparse import csgraph

from arcos4py import ARCOS
from arcos4py.tools import CoactivationReport
from arcos4py.tools import _coactivation_kernels as K
from arcos4py.tools import (
    calculate_statistics,
    calibrate_coactivation,
    detect_coactivation_events,
    detect_coactivation_events_image,
    filterCollev,
)
from arcos4py.tools._coactivation import _to_matrix, detect_matrix, null_threshold
from arcos4py.tools._coactivation_baseline import interp_columns, masked_baseline_activity
from arcos4py.tools._coactivation_graph import CellGraph

POS = ["x", "y"]


def _pulse(length, rise=2, plateau=3, decay=3.0):
    k = np.arange(length)
    g = np.where(k < rise, (k + 1) / rise, 1.0)
    return np.where(k >= rise + plateau, np.exp(-(k - rise - plateau + 1) / decay), g)


def simulate_sheet(
    seed=0, side=20, T=120, waves=((60.0, 60.0, 20), (140.0, 140.0, 70)), noise=0.1, spont=0.002, bleach=0.0
):
    """Cells on a jittered square lattice (spacing 10); circular waves (centre x, centre y, start frame) that
    spread one cell per frame; spontaneous pulses; Gaussian noise; optional exponential bleaching.
    Returns the long table and the ground-truth wave id per row (0 = background)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:side, 0:side]
    pos = np.c_[xx.ravel(), yy.ravel()] * 10.0 + rng.normal(0, 1.0, (side * side, 2))
    n = pos.shape[0]
    g = _pulse(25)
    act = np.zeros((T, n))
    gt = np.zeros((T, n), int)
    for w, (cx, cy, t0) in enumerate(waves, start=1):
        d = np.hypot(pos[:, 0] - cx, pos[:, 1] - cy) / 10.0
        for c in np.flatnonzero(d < 6):
            on = int(t0 + d[c])
            seg = g[: max(0, min(T - on, g.size))]
            act[on : on + seg.size, c] += seg
            gt[on : on + seg.size, c] = np.where(seg >= 0.5, w, gt[on : on + seg.size, c])
    for c, on in zip(rng.integers(0, n, rng.poisson(spont * n * T)), rng.integers(0, T, rng.poisson(spont * n * T))):
        seg = g[: max(0, min(T - on, g.size))]
        act[on : on + seg.size, c] += seg
    t = np.arange(T)[:, None]
    sig = (1.0 + act) * np.exp(-bleach * t / T) + noise * rng.normal(size=(T, n))
    tt, cc = np.meshgrid(np.arange(T), np.arange(n), indexing="ij")
    df = pd.DataFrame(
        {"t": tt.ravel(), "id": cc.ravel(), "x": pos[cc.ravel(), 0], "y": pos[cc.ravel(), 1], "m": sig.ravel()}
    )
    return df, gt.ravel()


@pytest.fixture(scope="module")
def sheet():
    return simulate_sheet()


# ----------------------------------------------------------------------------- kernels
def test_null_threshold():
    sizes = np.array([1, 2, 2, 3, 5, 8])
    # one null copy, one test, epsilon 1: larger than the largest chance group
    assert null_threshold(sizes, 1, 1, 1.0) == 9
    # four null copies: fewer than 4 chance groups >= s in total, i.e. s = 3 (the groups 8, 5 and 3)
    assert null_threshold(sizes, 4, 1, 1.0) == 3


def test_roll_within_lifetime_matches_numpy():
    rng = np.random.default_rng(1)
    D = rng.normal(size=(30, 6))
    obs = rng.random((30, 6)) > 0.3
    offs = rng.integers(0, 30, 6)
    out = K.roll_within_lifetime(D, obs, offs, np.nan)
    for j in range(6):
        idx = np.flatnonzero(obs[:, j])
        exp = np.full(30, np.nan)
        if idx.size:
            exp[idx] = np.roll(D[idx, j], offs[j] % idx.size)
        np.testing.assert_array_equal(out[:, j], exp)


def test_pulse_pairs_match_brute_force():
    rng = np.random.default_rng(2)
    n, T = 12, 40
    A = sparse.random(n, n, density=0.3, random_state=3, format="csr").astype(bool)
    A = ((A + A.T) > 0).astype(bool)
    A.setdiag(False)
    A.eliminate_zeros()
    A = A.tocsr()
    A.sort_indices()
    indptr = A.indptr.astype(np.int64)[None, :]
    indices = A.indices.astype(np.int64)
    cells = rng.integers(0, n, 60)
    times = rng.integers(0, T, 60)
    r, c = K.pulse_pairs(cells, times, n, indptr, indices, 3)
    got = set(zip(r.tolist(), c.tolist()))
    exp = {(p, q) for p in range(60) for q in range(60) if A[cells[p], cells[q]] and 0 <= times[q] - times[p] <= 3}
    assert got == exp
    hist = K.lag_histogram(cells, times, n, indptr, indices, 3)
    assert hist.sum() == len(exp)


def test_rolling_median_matches_pandas():
    rng = np.random.default_rng(4)
    for W in (5, 8, 51):
        X = np.round(rng.normal(size=(60, 4)), 1)
        X[rng.random(X.shape) < 0.3] = np.nan
        exp = pd.DataFrame(X).rolling(W, center=True, min_periods=1).median().to_numpy()
        np.testing.assert_array_equal(K.rolling_median_centered(X, W), exp)


def test_interpolation_matches_pandas():
    rng = np.random.default_rng(14)
    X = rng.normal(size=(40, 8))
    X[rng.random(X.shape) < 0.4] = np.nan
    X[:, 0] = np.nan
    np.testing.assert_array_equal(interp_columns(X, True), pd.DataFrame(X).interpolate(limit_area="inside").to_numpy())
    np.testing.assert_array_equal(
        interp_columns(X, False), pd.DataFrame(X).interpolate(limit_direction="both").to_numpy()
    )


def test_frame_cluster_sizes_match_scipy():
    rng = np.random.default_rng(5)
    n, T = 30, 5
    A = sparse.random(n, n, density=0.08, random_state=6, format="csr").astype(bool)
    A = ((A + A.T) > 0).tocsr()
    A.setdiag(False)
    A.eliminate_zeros()
    A.sort_indices()
    active = rng.random((T, n)) > 0.5
    got = np.sort(K.frame_cluster_sizes(active, A.indptr.astype(np.int64)[None, :], A.indices.astype(np.int64)))
    exp = []
    for t in range(T):
        act = np.flatnonzero(active[t])
        _, cl = csgraph.connected_components(A[act][:, act], directed=False)
        exp.append(np.bincount(cl))
    np.testing.assert_array_equal(got, np.sort(np.concatenate(exp)))


# ----------------------------------------------------------------------------- baseline
def test_baseline_removes_bleaching_and_keeps_pulses():
    df, gt = simulate_sheet(seed=7, bleach=1.0)
    T, n = df.t.max() + 1, df.id.max() + 1
    S = df.m.to_numpy().reshape(T, n)
    D, info = masked_baseline_activity(S, np.ones((T, n), bool))
    rest = gt.reshape(T, n) == 0
    assert abs(np.nanmedian(D[rest])) < 0.03  # flat resting level despite ~60 % bleaching
    assert np.nanmedian(D[gt.reshape(T, n) > 0]) > 0.3  # pulses stay in the data
    assert 51 <= info["window"] <= T


# ----------------------------------------------------------------------------- detection
def test_detects_waves(sheet):
    df, gt = sheet
    out, report = detect_coactivation_events(df, POS, "t", "id", "m", return_report=True)
    assert isinstance(report, CoactivationReport)
    assert report.n_events == out.collid.nunique() >= 2
    in_wave = gt > 0
    rows = df.assign(truth=gt).merge(out[["t", "id", "collid"]], on=["t", "id"], how="left")
    called = rows.collid.notna().to_numpy()
    assert called[in_wave].mean() > 0.6  # most wave cell-frames are found
    assert in_wave[called].mean() > 0.8  # and most calls are in waves
    # one event per wave: the two waves get different ids
    ids = [rows.collid[(rows.truth == w).to_numpy() & called].mode().iloc[0] for w in (1, 2)]
    assert ids[0] != ids[1]


def test_output_format_and_downstream(sheet):
    df, _ = sheet
    out = detect_coactivation_events(df, POS, "t", "id", "m", clid_column="clTrackID")
    assert list(out.columns) == list(df.columns) + ["clTrackID"]
    assert out.clTrackID.dtype == np.int64 and out.clTrackID.min() >= 1
    assert len(out.merge(df, on=list(df.columns))) == len(out)  # rows come from the input
    stats = calculate_statistics(out, frame_column="t", clid_column="clTrackID", position_columns=POS)
    assert len(stats) == out.clTrackID.nunique()
    filtered = filterCollev(out, frame_column="t", clid_column="clTrackID", obj_id_column="id").filter(
        min_duration=1, min_total_size=1
    )
    assert len(filtered) == len(out)


def test_no_waves_gives_at_most_epsilon_events():
    df, _ = simulate_sheet(seed=8, waves=())
    out = detect_coactivation_events(df, POS, "t", "id", "m")
    assert out.collid.nunique() <= 1


def test_deterministic(sheet):
    df, _ = sheet
    a = detect_coactivation_events(df, POS, "t", "id", "m", seed=3)
    b = detect_coactivation_events(df, POS, "t", "id", "m", seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_options_run(sheet):
    df, _ = sheet
    base = detect_coactivation_events(df, POS, "t", "id", "m")
    for kw in (
        dict(prune_early=False),
        dict(refine_start=False),
        dict(allow_merges=True),
        dict(remove_population_trend=False),
        dict(baseline_window=61),
    ):
        out, rep = detect_coactivation_events(df, POS, "t", "id", "m", return_report=True, **kw)
        assert out.collid.nunique() >= 1
    full = detect_coactivation_events(df, POS, "t", "id", "m", prune_early=False)
    assert len(full) >= len(base)  # pruning only removes members


def test_missing_frames_and_track_breaks(sheet):
    df, _ = sheet
    rng = np.random.default_rng(9)
    sub = df[rng.random(len(df)) > 0.05].copy()  # detections missing at random
    sub["id"] = sub.id * 10 + (sub.t >= 60)  # every track broken in two
    out = detect_coactivation_events(sub, POS, "t", "id", "m")
    assert out.collid.nunique() >= 1


def test_empty_input_raises(sheet):
    df, _ = sheet
    with pytest.raises(ValueError, match="Input is empty"):
        detect_coactivation_events(df.iloc[:0], POS, "t", "id", "m")


def test_arcos_class_method(sheet):
    df, _ = sheet
    ts = ARCOS(
        df, position_columns=POS, frame_column="t", obj_id_column="id", measurement_column="m", clid_column="collid"
    )
    a = ts.track_coactive_events()
    b = detect_coactivation_events(df, POS, "t", "id", "m")
    pd.testing.assert_frame_equal(
        a.sort_values(["t", "id"]).reset_index(drop=True), b.sort_values(["t", "id"]).reset_index(drop=True)
    )


def test_calibration_across_movies():
    movies = [simulate_sheet(seed=s)[0] for s in (10, 11)]
    calib = calibrate_coactivation(movies, POS, "t", "id", "m", n_null=4)
    assert calib.q in calib.deltas and calib.q_lo <= calib.q
    df, gt = simulate_sheet(seed=12)
    out, rep = detect_coactivation_events(df, POS, "t", "id", "m", calibration=calib, return_report=True)
    assert rep.level_quantile == calib.q
    assert out.collid.nunique() >= 1


def test_image_stack():
    rng = np.random.default_rng(13)
    T, Y, X = 80, 48, 48
    yy, xx = np.mgrid[0:Y, 0:X]
    stack = 1.0 + 0.05 * rng.normal(size=(T, Y, X))
    d = np.hypot(yy - 24, xx - 24)
    g = _pulse(20)
    for t in range(T):
        k = t - 20 - (d / 2).astype(int)  # a ring spreading 2 px per frame from frame 20
        ok = (k >= 0) & (k < g.size) & (d < 20)
        stack[t][ok] += g[k[ok]]
    lab, rep = detect_coactivation_events_image(stack, tile=4, mask=None, return_report=True)
    assert lab.shape == stack.shape and lab.dtype == np.int32
    assert rep.n_events >= 1
    assert (lab[:, d < 12] > 0).mean() > (lab[:, d > 22] > 0).mean()  # events where the ring is


def test_cellgraph_neighbours_are_local(sheet):
    df, _ = sheet
    T, n = df.t.max() + 1, df.id.max() + 1
    pos = df[POS].to_numpy().reshape(T, n, 2)
    g = CellGraph(pos, np.ones((T, n), bool))
    A = g.A_t[0].tocoo()
    dist = np.hypot(*(pos[0, A.row] - pos[0, A.col]).T)
    assert dist.max() < 25  # lattice spacing 10: no long Delaunay edges across the sheet


def test_cellgraph_3d_keeps_every_simplex_edge():
    rng = np.random.default_rng(15)
    zz, yy, xx = np.mgrid[0:6, 0:6, 0:6]
    pos = np.c_[xx.ravel(), yy.ravel(), zz.ravel()] * 10.0 + rng.normal(0, 1.0, (216, 3))
    g = CellGraph(pos[None], np.ones((1, 216), bool))
    A = g.A_t[0].tocsr()
    d = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    np.fill_diagonal(d, np.inf)
    nearest = d.argmin(1)  # the nearest neighbour is always a (short) Delaunay edge
    assert all(A[i, nearest[i]] for i in range(216))


def test_invalid_input_raises(sheet):
    df, _ = sheet
    with pytest.raises(ValueError, match="epsilon"):
        detect_coactivation_events(df, POS, "t", "id", "m", epsilon=0)
    with pytest.raises(ValueError, match="Duplicate"):
        detect_coactivation_events(pd.concat([df, df.iloc[:5]]), POS, "t", "id", "m")
    with pytest.raises(ValueError, match="2 position columns"):
        detect_coactivation_events(df, ["x"], "t", "id", "m")


def test_calibration_does_not_keep_null_copies():
    movies = [simulate_sheet(seed=s)[0] for s in (10, 11)]
    calib = calibrate_coactivation(movies, POS, "t", "id", "m", n_null=4)
    detect_coactivation_events(movies[0], POS, "t", "id", "m", calibration=calib)
    assert calib._cache is None and calib._cache_key is None


def test_frame_missing_for_every_cell_keeps_timing(sheet):
    df, _ = sheet
    full = detect_coactivation_events(df, POS, "t", "id", "m")
    gap = detect_coactivation_events(df[df.t != 50], POS, "t", "id", "m", return_report=True)[1]
    assert gap.n_events >= 1
    # frame values with a common spacing are not padded with empty frames
    spaced = detect_coactivation_events(df.assign(t=df.t * 5), POS, "t", "id", "m")
    assert len(spaced) == len(full)


def test_collinear_cells_do_not_crash():
    df, _ = simulate_sheet(seed=0, side=20, waves=((60.0, 0.0, 20),))
    line = df[df.id < 20].assign(y=0.0)  # one row of cells, exactly collinear
    out = detect_coactivation_events(line, POS, "t", "id", "m")
    assert isinstance(out, pd.DataFrame)


def test_more_invalid_input_raises(sheet):
    df, _ = sheet
    with pytest.raises(ValueError, match="n_null"):
        detect_coactivation_events(df, POS, "t", "id", "m", n_null=0)
    with pytest.raises(ValueError, match="at least one movie"):
        calibrate_coactivation([], POS, "t", "id", "m")
    with pytest.raises(ValueError, match="tile"):
        detect_coactivation_events_image(np.ones((10, 8, 8)), tile=0)


def test_calibration_with_a_quiet_movie():
    movies = [simulate_sheet(seed=10)[0], simulate_sheet(seed=11, waves=(), spont=0.0)[0]]
    calib = calibrate_coactivation(movies, POS, "t", "id", "m", n_null=4)
    assert set(calib.null_cellframes) == set(calib.deltas)


def test_arcos_class_method_clid_column_kwarg(sheet):
    df, _ = sheet
    ts = ARCOS(df, position_columns=POS, frame_column="t", obj_id_column="id", measurement_column="m")
    out = ts.track_coactive_events(clid_column="ev")
    assert "ev" in out.columns


def test_no_all_nan_warnings(sheet):
    import warnings

    df, _ = sheet
    sub = df.copy()
    sub.loc[sub.id == 0, "m"] = 5.0  # one cell that is never at rest
    with warnings.catch_warnings():
        warnings.filterwarnings("error", "All-NaN slice", RuntimeWarning)
        detect_coactivation_events(sub, POS, "t", "id", "m")


def test_smooth_k(sheet):
    df, _ = sheet
    base, rep3 = detect_coactivation_events(df, POS, "t", "id", "m", return_report=True)
    assert rep3.smooth_k == 3
    for k in (1, 5):
        out, rep = detect_coactivation_events(df, POS, "t", "id", "m", smooth_k=k, return_report=True)
        assert rep.smooth_k == k and out.collid.nunique() >= 2  # both waves still found
    for bad in (0, 2, 2.5):
        with pytest.raises(ValueError, match="smooth_k"):
            detect_coactivation_events(df, POS, "t", "id", "m", smooth_k=bad)


def test_calibration_remembers_smooth_k():
    movies = [simulate_sheet(seed=s)[0] for s in (10, 11)]
    calib = calibrate_coactivation(movies, POS, "t", "id", "m", n_null=4, smooth_k=5)
    assert calib.smooth_k == 5
    df, _ = simulate_sheet(seed=12)
    detect_coactivation_events(df, POS, "t", "id", "m", calibration=calib, smooth_k=5)
    with pytest.raises(ValueError, match="smooth_k=5"):
        detect_coactivation_events(df, POS, "t", "id", "m", calibration=calib)


def test_report_activity(sheet):
    df, gt = sheet
    shuffled = df.sample(frac=1.0, random_state=0)  # activity follows the index of X, not its order
    out, rep = detect_coactivation_events(shuffled, POS, "t", "id", "m", return_report=True)
    act = rep.activity
    assert isinstance(act, pd.Series) and act.index.equals(shuffled.index)
    assert act.notna().all()
    truth = pd.Series(gt, index=df.index).loc[shuffled.index]
    assert act[truth > 0].median() > 0.3 > abs(act[truth == 0].median())


def test_simultaneous_origins_are_separate_events():
    # three waves that start in the same frame ~10 cells apart and grow into each other: three events
    waves = ((50.0, 100.0, 30), (150.0, 100.0, 30), (100.0, 180.0, 30))
    df, gt = simulate_sheet(seed=0, waves=waves, T=110)
    out = detect_coactivation_events(df, POS, "t", "id", "m")
    rows = df.assign(truth=gt).merge(out[["t", "id", "collid"]], on=["t", "id"], how="left")
    ids = [rows.collid[(rows.truth == w).to_numpy() & rows.collid.notna().to_numpy()].mode().iloc[0] for w in (1, 2, 3)]
    assert len(set(ids)) == 3


def test_every_event_has_a_single_origin(sheet):
    from arcos4py.tools._coactivation import _near

    df, _ = sheet
    S, observed, pos_t, rows = _to_matrix(df, POS, "t", "id", "m")
    g = CellGraph(pos_t, observed)
    lab, _ = detect_matrix(S, observed, g)
    for e in range(1, lab.max() + 1):
        t0 = np.flatnonzero((lab == e).any(1))[0]
        first = np.flatnonzero(lab[t0] == e)
        k, _ = csgraph.connected_components(_near(g, t0)[first][:, first], directed=False)
        assert k == 1  # the event appears as one piece


def test_single_origin_drops_parts_before_contact():
    from arcos4py.tools._coactivation import single_origin
    from arcos4py.tools._coactivation_graph import GridGraph

    keep = np.argwhere(np.ones((1, 12), bool))  # a row of 12 tiles: 0 - 1 - ... - 11
    g = GridGraph(keep, 4)
    lab = np.zeros((4, 12), np.int32)
    lab[0, 0] = lab[1, :2] = 1  # the origin grows from tile 0
    lab[1, 9:] = 1  # a separate patch far away, before the front reaches it
    lab[2, :6] = lab[2, 9:] = 1  # still separate
    lab[3, :] = 1  # the front reaches it
    out = single_origin(lab, g)
    assert (out[1, 9:] == 0).all() and (out[2, 9:] == 0).all()
    assert (out[3] == 1).all() and (out[1, :2] == 1).all()
