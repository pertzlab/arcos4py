"""Collective events from continuous single-cell signals, without binarisation.

An event is a group of neighbouring units (cells, or tiles of an image) whose signal rises together, more
than the same traces rise together after shifting each unit in time. Every threshold is calibrated against
such time-shifted copies of the data (the null); the only user parameter is epsilon, the number of false
events one is willing to expect per movie.

Steps:

1. Activity dR per unit and frame: smooth_k-frame mean (default 3) minus a resting-frame baseline (shared drift such as
   bleaching, plus each unit's own slow drift; see _coactivation_baseline).
2. Co-activation score L = min(own dR, mean dR of the unit's neighbours in that frame). High only if the
   unit and its neighbourhood are raised together.
3. Pulses: runs of frames with L above a level h, h a quantile of L in the null. A pulse's time is the
   steepest rise of dR around its start.
4. Onset groups: pulses of neighbouring units are linked if their times differ by at most delta frames;
   delta is the largest neighbour lag at which such pairs are at least twice as common as in time-shifted
   copies of the calls. Optionally a density core m: a pulse is core if linked to at least m others.
5. Significance: a group is an event if it is larger than chance groups (same pipeline on the null) are
   expected to be in fewer than epsilon movies, over a ladder of levels and cores (Bonferroni). The level
   and core with the most events are kept (ties: the most stringent).
6. Growth: events are grown into weaker co-active pulses (lowest level) that are onset-linked to them.
7. Membership: full width at half maximum of each event (per-event threshold, per-cell runs).
8. Births: inside every event, units are tracked frame by frame (ARCOS-style); activations that started
   separately and are large enough to be more than chance stay separate events.
9. Start of membership read on the unsmoothed dR (the smoothing delays sharp rises).
10. Early outliers removed: a member that started more than k frames before every neighbouring member of
    its event (k: the lag at which neighbour pairs are still >= 2.5x as common as by chance) is most often a
    pulse of its own that happened to touch the wave.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph

from . import _coactivation_kernels as K
from ._coactivation_baseline import masked_baseline_activity, pulses_from_binary, smoothed
from ._coactivation_graph import CellGraph

LEVEL_QUANTILES = (0.95, 0.97, 0.98, 0.99, 0.995)
CORE_LADDER = (0, 1, 2, 3)
MAX_LAG = 8
GROW_STEPS = 2
N_PREV = 2
LINK_K = 1
MEMBERSHIP_FRAC = 0.5
STRICT_RATIO = 2.5


# ============================================================================ null calibration
def null_threshold(null_sizes, n_null, n_tests, epsilon):
    """Smallest s with n_tests * (#null sizes >= s) / n_null < epsilon."""
    allowed = int(np.ceil(epsilon * n_null / n_tests)) - 1
    if null_sizes.size <= allowed:
        return 1
    desc = np.sort(null_sizes)[::-1]
    return int(desc[allowed]) + 1


def null_threshold_tail(null_sizes, n_null, n_tests, epsilon):
    """As null_threshold, with the upper tail of the null sizes extrapolated.

    The tail is an exponential fit to the largest 5 % (at least 10) groups, so that epsilon / n_tests can be
    resolved with few null copies; never below the empirical rule.
    """
    emp = null_threshold(null_sizes, n_null, n_tests, epsilon)
    if null_sizes.size < 10 or epsilon * n_null / n_tests >= 1:
        return emp
    desc = np.sort(null_sizes)[::-1].astype(float)
    k = max(10, int(0.05 * desc.size))
    u = desc[k - 1]
    excess = desc[:k] - u
    if excess.mean() <= 0:
        return emp
    lam = 1.0 / excess.mean()
    s = u + np.log(n_tests * k / (n_null * epsilon)) / lam
    return int(max(emp, np.floor(s) + 1))


def _null_offsets(observed, rng):
    """One random roll offset per unit (within its observed frames), drawn unit by unit."""
    counts = observed.sum(0)
    return np.array([rng.integers(m) if m else 0 for m in counts], np.int64)


def shift_within_lifetime(D, observed, rng):
    """Null copy: every unit's dR circularly shifted along its own observed frames."""
    return K.roll_within_lifetime(np.ascontiguousarray(D, np.float64), observed, _null_offsets(observed, rng), np.nan)


def local_score(D, observed, graph, cap):
    """L = min(own dR, mean dR of the neighbours present in that frame); -inf where undefined."""
    T, n = D.shape
    L = np.full((T, n), -np.inf)
    Dc = np.where(observed, np.minimum(np.nan_to_num(D, nan=0.0), cap), 0.0)
    for t in range(T):
        A = graph.A_t[t]
        if A.nnz == 0:
            continue
        cnt = A @ observed[t].astype(float)
        s = A @ Dc[t]
        ok = observed[t] & (cnt > 0)
        L[t, ok] = np.minimum(D[t, ok], s[ok] / cnt[ok])
    return L


# ============================================================================ pulses and onset groups
def _rise(D):
    return np.vstack([np.zeros((1, D.shape[1])), np.diff(np.nan_to_num(D), axis=0)])


def _timing(cells, onsets, offsets, R):
    if cells.size == 0:
        return np.zeros(0, np.int64)
    return K.pulse_timing(
        np.ascontiguousarray(cells, np.int64),
        np.ascontiguousarray(onsets, np.int64),
        np.ascontiguousarray(offsets, np.int64),
        np.ascontiguousarray(R, np.float64),
    )


def _pulse_graph(cells, times, graph, delta):
    """Pulses of neighbouring units (graph of the earlier pulse's frame) with times at most delta apart."""
    r, c = K.pulse_pairs(cells, times, graph.n, graph.indptr, graph.indices, delta)
    G = sparse.coo_matrix((np.ones(2 * r.size, bool), (np.r_[r, c], np.r_[c, r])), shape=(cells.size,) * 2)
    return G.tocsr()


def _pulse_components(cells, times, graph, delta, m=0):
    """Group per pulse (-1 = none).

    m = 0: connected components of the pulse graph. m >= 1: DBSCAN-style, a pulse is core if linked to >= m
    others; cores connect through core-core links, a non-core pulse linked to a core joins it, other pulses
    are in no group.
    """
    G = _pulse_graph(cells, times, graph, delta)
    if m <= 0:
        return csgraph.connected_components(G, directed=False)[1]
    deg = np.diff(G.indptr)
    core = deg >= m
    lab = np.full(cells.size, -1)
    if not core.any():
        return lab
    ci = np.flatnonzero(core)
    _, cc = csgraph.connected_components(G[ci][:, ci], directed=False)
    lab[ci] = cc
    border = np.flatnonzero(~core & (deg > 0))
    if border.size:
        Gb = G[border][:, ci].tocsr()
        has = np.diff(Gb.indptr) > 0
        lab[border[has]] = cc[Gb.indices[Gb.indptr[:-1][has]]]
    return lab


def _lag_counts(cells, times, graph, max_lag):
    return K.lag_histogram(cells, times, graph.n, graph.indptr, graph.indices, max_lag)


def _sizes(comp):
    comp = comp[comp >= 0]
    return np.bincount(comp) if comp.size else np.zeros(0, int)


def _strict_window(real, mu, z, ratio, delta):
    ok = (real >= ratio * mu) & (z > 3)
    return min(int(np.flatnonzero(ok).max()) if ok.any() else 0, delta)


# ============================================================================ births and merges
def spacetime_edges(graph, observed):
    """Edges between unit-frames: neighbours in one frame, and the same unit in consecutive frames."""
    T, n = observed.shape
    if graph.static_A is not None:
        c = sparse.triu(graph.static_A, k=1).tocoo()
        tt = np.repeat(np.arange(T, dtype=np.int64), c.nnz)
        u0 = tt * n + np.tile(c.row, T)
        v0 = tt * n + np.tile(c.col, T)
        keep = observed.ravel()[u0] & observed.ravel()[v0]
        us, vs = [u0[keep]], [v0[keep]]
    else:
        us, vs = [], []
        for t, A in enumerate(graph.A_t):
            c = sparse.triu(A, k=1).tocoo()
            us.append(t * n + c.row)
            vs.append(t * n + c.col)
    both = observed[:-1] & observed[1:]
    t_, c_ = np.nonzero(both)
    us.append(t_ * n + c_)
    vs.append((t_ + 1) * n + c_)
    return np.concatenate(us).astype(np.int64), np.concatenate(vs).astype(np.int64)


def _null_seed_size(graph, null_L, observed, h, epsilon):
    """Largest co-active patch (connected neighbours above h in one frame) chance produces."""
    sizes = [K.frame_cluster_sizes(observed & (Ln > h), graph.indptr, graph.indices) for Ln in null_L]
    sizes = np.concatenate(sizes) if sizes else np.zeros(0, int)
    return null_threshold(sizes, max(len(null_L), 1), 1, epsilon)


def _track_seeded(graph, binary, seedable, seed_size):
    """Frame-by-frame tracking (link to labelled units within 2*LINK_K+1 hops of the last N_PREV frames).

    An unlinked cluster starts a new birth only if it holds >= seed_size seedable (co-active) unit-frames.
    """
    return K.track_seeded(
        binary, seedable, graph.observed, graph.indptr, graph.indices, seed_size, N_PREV, 2 * LINK_K + 1
    )


def _join_simultaneous_births(sub, flat, e, eu, ev, n, delta):
    """Join births of event e that start and touch close together in time.

    Births that start within delta frames of each other and touch within delta frames of the later start are
    one start (patches of one front, or a pulse of the whole field).
    """
    T = sub.shape[0]
    nb = int(sub.max()) + 1
    start = np.full(nb, T)
    tt, ii = np.nonzero(sub)
    np.minimum.at(start, sub[tt, ii], tt)
    sf = sub.ravel()
    inside = (flat[eu] == e) & (flat[ev] == e)
    a, b = sf[eu[inside]], sf[ev[inside]]
    fr = np.minimum(eu[inside], ev[inside]) // n
    k = (a > 0) & (b > 0) & (a != b)
    a, b, fr = a[k], b[k], fr[k]
    close = (fr - np.maximum(start[a], start[b]) <= delta) & (np.abs(start[a] - start[b]) <= delta)
    if not close.any():
        return sub
    G = sparse.coo_matrix((np.ones(close.sum()), (a[close], b[close])), shape=(nb, nb))
    _, comp = csgraph.connected_components(G, directed=False)
    first = {}
    for lbl in np.argsort(start):
        first.setdefault(comp[lbl], lbl)
    lut = np.array([first[comp[x]] for x in range(nb)])
    lut[0] = 0
    return lut[sub]


def separate_births(lab, graph, eu, ev, seedable, min_cells, delta):
    """Inside every event, activations that started separately and reach >= min_cells units stay separate.

    Smaller births and unlinked unit-frames join the nearest large birth (space-time distance).
    """
    T, n = lab.shape
    out = lab.copy()
    next_id = int(lab.max()) + 1
    flat = lab.ravel()
    for e in np.unique(lab[lab > 0]):
        mask = lab == e
        sub = np.where(mask, _track_seeded(graph, mask, seedable & mask, min_cells), 0)
        if sub.max() > 1:
            sub = _join_simultaneous_births(sub, flat, e, eu, ev, n, delta)
        n_cells = np.array([np.unique(np.nonzero(sub == k)[1]).size for k in range(sub.max() + 1)])
        big = np.flatnonzero(n_cells >= min_cells)
        big = big[big > 0]
        if big.size <= 1:
            continue
        nodes = np.flatnonzero(mask.ravel())
        idx = np.full(T * n, -1)
        idx[nodes] = np.arange(nodes.size)
        inside = (flat[eu] == e) & (flat[ev] == e)
        G = sparse.coo_matrix(
            (np.ones(inside.sum()), (idx[eu[inside]], idx[ev[inside]])), shape=(nodes.size,) * 2
        ).tocsr()
        sf = sub.ravel()[nodes]
        dist = np.vstack(
            [
                csgraph.dijkstra(G, directed=False, unweighted=True, indices=np.flatnonzero(sf == b), min_only=True)
                for b in big
            ]
        )
        owner = big[np.argmin(dist, axis=0)]
        owner = np.where(np.isin(sf, big), sf, owner)
        largest = big[np.argmax(n_cells[big])]
        new_ids = {int(b): (int(e) if b == largest else next_id + k) for k, b in enumerate(big)}
        next_id += big.size
        o = out.ravel()
        o[nodes] = np.array([new_ids[int(x)] for x in owner])
        out = o.reshape(T, n)
    return out


def event_contacts(lab, group, eu, ev, n):
    """Pairs of events that touch.

    Columns: first contact frame, number of contacts, and whether both came out of one onset group (two
    separately started waves that collided = a merge).
    """
    o, g = lab.ravel(), group.ravel()
    a, b = o[eu], o[ev]
    m = (a > 0) & (b > 0) & (a != b)
    cols = ["event_a", "event_b", "first_contact", "contacts", "merge"]
    if not m.any():
        return pd.DataFrame(columns=cols)
    grp = {}
    for e in np.unique(o[o > 0]):
        v = g[o == e]
        v = v[v > 0]
        grp[int(e)] = int(np.bincount(v).argmax()) if v.size else -int(e)
    pa, pb = np.minimum(a[m], b[m]), np.maximum(a[m], b[m])
    C = pd.DataFrame({"event_a": pa, "event_b": pb, "frame": np.minimum(eu[m], ev[m]) // n})
    C = C.groupby(["event_a", "event_b"]).frame.agg(first_contact="min", contacts="size").reset_index()
    C["merge"] = [grp[int(x)] == grp[int(y)] for x, y in zip(C.event_a, C.event_b)]
    return C[cols]


def apply_merges(lab, contacts):
    """Two events that came out of one onset group continue as one new event from their first contact on.

    Returns the labels and a lineage table (event, parent_a, parent_b, frame).
    """
    lab = lab.copy()
    next_id = int(lab.max()) + 1
    rows = []
    if contacts is None or contacts.empty:
        return lab, pd.DataFrame(columns=["event", "parent_a", "parent_b", "frame"])
    M = contacts[contacts["merge"]].sort_values("first_contact")
    orig = lab.copy()
    for a, b, t in zip(M.event_a, M.event_b, M.first_contact):
        ca = lab[t][orig[t] == a]
        cb = lab[t][orig[t] == b]
        if ca.size == 0 or cb.size == 0:
            continue
        la, lb = np.bincount(ca).argmax(), np.bincount(cb).argmax()
        if la == lb:
            continue
        sel = np.isin(lab[t:], [la, lb])
        lab[t:][sel] = next_id
        rows.append(dict(event=next_id, parent_a=int(la), parent_b=int(lb), frame=int(t)))
        next_id += 1
    return lab, pd.DataFrame(rows, columns=["event", "parent_a", "parent_b", "frame"])


# ============================================================================ membership refinements
def halfmax_members(lab, D, observed, frac=MEMBERSHIP_FRAC):
    """Full width at half maximum membership.

    Per event, theta = frac x the 95th percentile of dR over its unit-frames; unit-frames below theta leave
    the event and every member keeps the event id along its run above theta.
    """
    ne = int(lab.max())
    if ne == 0:
        return lab
    Dz = np.where(observed, np.nan_to_num(D, nan=-np.inf), -np.inf)
    th = np.full(ne + 1, np.inf)
    tt, cc = np.nonzero(lab)
    e = lab[tt, cc]
    v = Dz[tt, cc]
    order = np.argsort(e, kind="stable")
    e_s, v_s = e[order], v[order]
    bounds = np.flatnonzero(np.r_[True, e_s[1:] != e_s[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        th[e_s[a]] = frac * np.quantile(v_s[a:b], 0.95)
    out = np.where(Dz > th[lab], lab, 0).astype(np.int32)
    out[lab == 0] = 0
    return K.halfmax_extend(out, lab.astype(np.int32), Dz, observed, th)


def _member_items(lab, values=None):
    """(event, unit) items sorted by event then unit: first frame and, optionally, the max of `values`."""
    n = lab.shape[1]
    tt, cc = np.nonzero(lab)
    e = lab[tt, cc].astype(np.int64)
    uniq, inv = np.unique(e * n + cc, return_inverse=True)
    s0 = np.full(uniq.size, np.iinfo(np.int64).max)
    np.minimum.at(s0, inv, tt)
    pk = None
    if values is not None:
        pk = np.full(uniq.size, -np.inf)
        np.fmax.at(pk, inv, values[tt, cc])
        pk[~np.isfinite(pk)] = np.nan
    return uniq // n, uniq % n, s0, pk


def refine_starts(lab, D, D_raw, observed, frac=MEMBERSHIP_FRAC, reach=2):
    """Each member's first frame read on the unsmoothed dR.

    Within +-reach frames of its current start, the first frame where the unsmoothed dR reaches frac x the
    member's own peak (smoothed) dR.
    """
    if lab.max() == 0:
        return lab
    ie, ic, s0, pk = _member_items(lab, D)
    return K.refine_starts(
        lab.astype(np.int32).copy(),
        ie.astype(np.int32),
        ic,
        s0,
        frac * pk,
        np.ascontiguousarray(D_raw),
        observed,
        reach,
    )


def prune_early_members(lab, graph, k):
    """Remove each member that started more than k frames before every neighbouring member of its event.

    Members without any neighbouring member are removed as well.
    """
    if lab.max() == 0:
        return lab
    ie, ic, s0, _ = _member_items(lab)
    start_of = np.full((int(lab.max()) + 1, lab.shape[1]), -1, np.int64)
    start_of[ie, ic] = s0
    bad = K.early_outliers(ie, ic, s0, start_of, graph.indptr, graph.indices, k)
    out = lab.copy()
    for e, c in zip(ie[bad], ic[bad]):
        col = out[:, c]
        col[col == e] = 0
    return out


def _relabel(lab):
    u, inv = np.unique(lab, return_inverse=True)
    if u[0] != 0:
        inv = inv + 1
    return inv.reshape(lab.shape).astype(np.int32)


# ============================================================================ calibration across movies
@dataclass
class CoactivationCalibration:
    """Frozen event definition from several movies (see calibrate_coactivation).

    Attributes:
        q (float): co-activation level of the events (null quantile).
        q_lo (float): growth level (null quantile).
        m (int): density core.
        levels (dict): levels in signal units (median over the calibration movies).
        deltas (dict): onset window per level.
        strict_deltas (dict): window for removing early members, per level.
        null_sizes (dict): pooled null group sizes per (level, core).
        null_cellframes (dict): unit-frames behind the pooled null sizes, per level.
        epsilon (float): expected false events per movie.
        level_mode (str): "quantile" (each movie gets its level from its own null quantile) or "absolute".
        smooth_k (int): temporal smoothing window the calibration was made with (detection must use the same).
    """

    q: float
    q_lo: float
    m: int
    levels: dict
    deltas: dict
    strict_deltas: dict
    null_sizes: dict
    null_cellframes: dict
    epsilon: float = 1.0
    level_mode: str = "quantile"
    smooth_k: int = 3
    _cache: list | None = field(default=None, repr=False)
    _cache_key: int | None = field(default=None, repr=False)

    def threshold(self, q, m, n_cellframes, n_tests=1):
        """Smallest group size with fewer than epsilon chance groups expected in a movie of this size."""
        n_eff = self.null_cellframes[q] / max(n_cellframes, 1)
        return null_threshold_tail(np.asarray(self.null_sizes[(q, m)]), n_eff, n_tests, self.epsilon)

    def own_null(self, D, observed, graph, cap, rng, n_copies=2):
        """A few null copies of the movie being analysed (cached)."""
        key = id(D)
        if self._cache_key != key:
            self._cache = [
                (Dn, local_score(Dn, observed, graph, cap))
                for Dn in (shift_within_lifetime(D, observed, rng) for _ in range(n_copies))
            ]
            self._cache_key = key
        return self._cache

    def level_for(self, q, D, observed, graph, cap, rng):
        """Co-activation level for quantile q in this movie's signal units."""
        if self.level_mode == "absolute":
            return float(self.levels[q])
        v = np.concatenate([Ln[observed] for _, Ln in self.own_null(D, observed, graph, cap, rng)])
        return float(np.quantile(v[np.isfinite(v)], q))


# ============================================================================ detection on matrices
@dataclass
class _Movie:
    """Everything the detection steps share for one movie."""

    S: np.ndarray
    observed: np.ndarray
    graph: object
    D: np.ndarray
    L: np.ndarray
    R: np.ndarray
    cap: float
    rng: np.random.Generator
    nulls: list
    epsilon: float
    n_null: int
    calibration: CoactivationCalibration | None
    calib_data: dict


def _prepare(
    S,
    observed,
    graph,
    epsilon,
    n_null,
    seed,
    remove_population_trend,
    baseline_window,
    calibration,
    level_quantiles,
    core_ladder,
    smooth_k,
):
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():  # units without resting frames give all-NaN medians; they are handled
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        D, binfo = masked_baseline_activity(
            S, observed, population=remove_population_trend, window=baseline_window, smooth=smooth_k
        )
    cap = np.nanquantile(D[observed], 0.99)
    L = local_score(D, observed, graph, cap)
    nulls = []
    if calibration is None:
        for _ in range(n_null):
            Dn = shift_within_lifetime(D, observed, rng)
            nulls.append((Dn, local_score(Dn, observed, graph, cap)))
        null_vals = np.concatenate([Ln[observed] for _, Ln in nulls])
        null_vals = null_vals[np.isfinite(null_vals)]
        level_list = [(q_, float(np.quantile(null_vals, q_))) for q_ in level_quantiles]
    else:
        level_list = [
            (q_, calibration.level_for(q_, D, observed, graph, cap, rng))
            for q_ in sorted({calibration.q, calibration.q_lo})
        ]
        core_ladder = (calibration.m,)
    calib_data = {"n_obs": int(observed.sum()), "n_null": n_null, "levels": {}}
    mv = _Movie(S, observed, graph, D, L, _rise(D), cap, rng, nulls, epsilon, n_null, calibration, calib_data)
    return mv, level_list, core_ladder, binfo


def _evaluate_calibrated(mv, q, h, cells, on):
    """Groups at level q with the calibration's window, core and size threshold.

    The threshold is never below this movie's own largest chance group.
    """
    c = mv.calibration
    delta, m = int(c.deltas[q]), c.m
    comp = _pulse_components(cells, on, mv.graph, delta, m)
    sz = _sizes(comp)
    thr = c.threshold(q, m, mv.calib_data["n_obs"])
    own = []
    for Dn, Ln in c.own_null(mv.D, mv.observed, mv.graph, mv.cap, mv.rng):
        c_, a_, b_ = pulses_from_binary(mv.observed & (Ln > h))
        own.append(_sizes(_pulse_components(c_, _timing(c_, a_, b_, _rise(Dn)), mv.graph, delta, m)))
    own = np.concatenate(own) if own else np.zeros(0, int)
    if own.size:
        thr = max(thr, int(own.max()) + 1)
    keep = sz >= thr
    row = dict(q=q, h=h, delta=delta, m=m, min_pulses=thr, n_events=int(keep.sum()))
    return delta, {m: thr}, [(row, comp, keep)]


def _onset_window(mv, C, cells, on):
    """Largest neighbour lag at which pulse pairs are >= 2x as common as after shifting the calls in time."""
    counts = mv.observed.sum(0)
    lag_null = []
    for _ in range(mv.n_null):
        offs = np.array([mv.rng.integers(c) if c else 0 for c in counts], np.int64)
        Cs = K.roll_within_lifetime(C.astype(np.float64), mv.observed, offs, 0.0) > 0.5
        Rs = K.roll_within_lifetime(mv.R, mv.observed, offs, 0.0)
        c_, a_, b_ = pulses_from_binary(Cs)
        lag_null.append(_lag_counts(c_, _timing(c_, a_, b_, Rs), mv.graph, MAX_LAG))
    real = _lag_counts(cells, on, mv.graph, MAX_LAG)
    nl = np.array(lag_null)
    mu = nl.mean(0)
    z = (real - mu) / (nl.std(0) + np.sqrt(mu + 1))
    ok = (real >= 2 * mu) & (z > 3)
    return (int(np.flatnonzero(ok).max()) if ok.any() else 0), real, nl, mu, z


def _evaluate_free(mv, q, h, C, cells, on, core_ladder, n_tests, fixed_deltas):
    """Onset window from this movie's lag statistics and size thresholds from its own null copies."""
    delta, real, nl, mu, z = _onset_window(mv, C, cells, on)
    if fixed_deltas is not None:
        delta = int(fixed_deltas[q])
    cd = mv.calib_data["levels"][q] = dict(h=h, lag_real=real, lag_null=nl, sizes={}, null_sizes={})
    npul = []
    for Dn, Ln in mv.nulls:
        c_, a_, b_ = pulses_from_binary(mv.observed & (Ln > h))
        npul.append((c_, _timing(c_, a_, b_, _rise(Dn))))
    thrs, out = {}, []
    for m in core_ladder:
        comp = _pulse_components(cells, on, mv.graph, delta, m)
        sz = _sizes(comp)
        nsz = np.concatenate([_sizes(_pulse_components(c_, o_, mv.graph, delta, m)) for c_, o_ in npul])
        thr = thrs[m] = null_threshold_tail(nsz, mv.n_null, n_tests, mv.epsilon)
        cd["sizes"][m], cd["null_sizes"][m] = sz, nsz
        keep = sz >= thr
        out.append(
            (
                dict(q=q, h=h, delta=delta, m=m, min_pulses=thr, n_events=int(keep.sum()), n_pulses=int(cells.size)),
                comp,
                keep,
            )
        )
    return delta, thrs, out, dict(real=real, mu=mu, z=z)


def _select_level(mv, level_list, core_ladder, n_tests, fixed_deltas):
    """Evaluate every level (and core); keep the one with the most events (ties: most stringent)."""
    best, table, lvl = None, [], {}
    for q, h in level_list:
        if fixed_deltas is not None and q not in fixed_deltas:
            continue  # no onset window for this level (no pulses there in the calibration's first pass)
        C = mv.observed & (mv.L > h)
        cells, on_bin, off = pulses_from_binary(C)
        if cells.size == 0:
            continue
        on = _timing(cells, on_bin, off, mv.R)
        lag = {}
        if mv.calibration is not None:
            delta, thrs, rows = _evaluate_calibrated(mv, q, h, cells, on)
        else:
            delta, thrs, rows, lag = _evaluate_free(mv, q, h, C, cells, on, core_ladder, n_tests, fixed_deltas)
        lvl[q] = dict(h=h, cells=cells, on_bin=on_bin, off=off, on=on, delta=delta, thr=thrs, **lag)
        for row, comp, keep in rows:
            table.append(row)
            if mv.calibration is not None:
                if q == mv.calibration.q:
                    best = ((row["n_events"], 0.0), q, row["m"], comp, keep)
            elif best is None or (row["n_events"], q, row["m"]) > best[0]:
                best = ((row["n_events"], q, row["m"]), q, row["m"], comp, keep)
    return best, table, lvl


def _grow(mv, lvl, q, m, ev_of):
    """Events grown into the onset-linked co-active pulses of the lowest level (hysteresis).

    Groups there that touch no event become events if they pass that level's own test.
    """
    T, n = mv.observed.shape
    hi, lo = lvl[q], lvl[min(lvl)]
    pid = np.full((T, n), -1)
    for k_, (c, a, b_) in enumerate(zip(lo["cells"], lo["on_bin"], lo["off"])):
        pid[a:b_, c] = k_
    lab_lo = np.full(lo["cells"].size, -1)
    for c, a, e in zip(hi["cells"], hi["on_bin"], ev_of):
        if e >= 0 and lab_lo[pid[a, c]] < 0:
            lab_lo[pid[a, c]] = e
    G = _pulse_graph(lo["cells"], lo["on"], mv.graph, lo["delta"])
    n_ev0 = int(ev_of.max()) + 1
    for _ in range(GROW_STEPS):
        un = np.flatnonzero(lab_lo < 0)
        Gu = G[un].tocoo()
        tgt = lab_lo[Gu.col]
        ok = tgt >= 0
        if not ok.any():
            break
        votes = sparse.coo_matrix((np.ones(ok.sum()), (Gu.row[ok], tgt[ok])), shape=(un.size, n_ev0)).tocsr()
        has = np.diff(votes.indptr) > 0
        lab_lo[un[has]] = np.asarray(votes[has].argmax(axis=1)).ravel()
    un = np.flatnonzero(lab_lo < 0)
    if un.size:
        comp_u = _pulse_components(lo["cells"][un], lo["on"][un], mv.graph, lo["delta"], m)
        for k_, b in enumerate(np.flatnonzero(_sizes(comp_u) >= lo["thr"][m])):
            lab_lo[un[comp_u == b]] = n_ev0 + k_
    return lo["cells"], lo["on_bin"], lo["off"], lab_lo


def _prune_window(mv, lvl, q, delta):
    if "real" in lvl[q]:
        return _strict_window(lvl[q]["real"], lvl[q]["mu"], lvl[q]["z"], STRICT_RATIO, delta)
    if mv.calibration is not None and q in mv.calibration.strict_deltas:
        return min(int(mv.calibration.strict_deltas[q]), delta)
    return delta


def _relabel_lineage(lineage, u):
    if lineage is None or not len(lineage):
        return lineage
    lut = dict(zip(u.tolist(), range(0 if u[0] == 0 else 1, u.size + (0 if u[0] == 0 else 1))))
    return lineage.assign(**{c: lineage[c].map(lambda v: lut.get(v, -1)) for c in ["event", "parent_a", "parent_b"]})


def detect_matrix(
    S,
    observed,
    graph,
    epsilon=1.0,
    n_null=8,
    seed=0,
    remove_population_trend=True,
    baseline_window=None,
    allow_merges=False,
    refine_start=True,
    prune_early=True,
    calibration=None,
    level_quantiles=LEVEL_QUANTILES,
    core_ladder=CORE_LADDER,
    fixed_deltas=None,
    return_calibration_data=False,
    smooth_k=3,
):
    """Co-activation events on a (T, n) signal matrix with its neighbour graph. Returns (labels (T, n), report)."""
    T, n = observed.shape
    _check_epsilon(epsilon)
    if int(n_null) < 1:
        raise ValueError(f"n_null must be >= 1, got {n_null}")
    if int(smooth_k) != smooth_k or smooth_k < 1 or smooth_k % 2 == 0:
        raise ValueError(f"smooth_k must be an odd number of frames >= 1 (1 = no smoothing), got {smooth_k}")
    smooth_k = int(smooth_k)
    if calibration is not None and calibration.smooth_k != smooth_k:
        raise ValueError(
            f"The calibration was made with smooth_k={calibration.smooth_k}; detect with the same smooth_k"
        )
    if calibration is not None:  # the null-copy cache belongs to one movie; never reuse it across calls
        calibration._cache, calibration._cache_key = None, None
    mv, level_list, core_ladder, binfo = _prepare(
        S,
        observed,
        graph,
        epsilon,
        n_null,
        seed,
        remove_population_trend,
        baseline_window,
        calibration,
        level_quantiles,
        core_ladder,
        smooth_k,
    )
    best, table, lvl = _select_level(mv, level_list, core_ladder, len(level_quantiles) * len(core_ladder), fixed_deltas)
    report = CoactivationReport(table=pd.DataFrame(table), baseline_window=binfo["window"], smooth_k=smooth_k)
    report.calibration_data = mv.calib_data if return_calibration_data else None
    if best is None:
        return np.zeros((T, n), np.int32), report
    _, q, m, comp, keep = best
    h, delta, thr = lvl[q]["h"], lvl[q]["delta"], lvl[q]["thr"][m]
    ev_of = np.full(comp.size, -1)
    in_group = comp >= 0
    if keep.size:
        ev_of[in_group] = np.where(keep[comp[in_group]], comp[in_group], -1)
    if min(lvl) < q:
        cells, on_bin, off, ev_of = _grow(mv, lvl, q, m, ev_of)
    else:
        cells, on_bin, off = lvl[q]["cells"], lvl[q]["on_bin"], lvl[q]["off"]
    lab = np.zeros((T, n), np.int32)
    for c, a, b_, e in zip(cells, on_bin, off, ev_of):
        if e >= 0:
            lab[a:b_, c] = e + 1
    group = lab.copy()  # onset groups, before births split them
    lab = halfmax_members(lab, mv.D, observed)
    # births: activations that started separately stay separate events
    eu, ev = spacetime_edges(graph, observed)
    null_L = [Ln for _, Ln in (mv.nulls if mv.nulls else calibration.own_null(mv.D, observed, graph, mv.cap, mv.rng))]
    seed_size = _null_seed_size(graph, null_L, observed, h, epsilon)
    lab = separate_births(lab, graph, eu, ev, observed & (mv.L > h), seed_size, delta)
    lineage = None
    if allow_merges:
        lab, lineage = apply_merges(lab, event_contacts(lab, group, eu, ev, n))
    if refine_start and lab.max() > 0:
        lab = refine_starts(lab, mv.D, mv.D + (smoothed(S, observed, 1) - smoothed(S, observed, smooth_k)), observed)
    prune_k = _prune_window(mv, lvl, q, delta) if prune_early else None
    if prune_k is not None:
        lab = prune_early_members(lab, graph, prune_k)
    u = np.unique(lab)
    lab = _relabel(lab)
    report.level_quantile, report.level, report.onset_window, report.core = q, h, delta, m
    report.min_event_size, report.seed_size, report.prune_window = int(thr), int(seed_size), prune_k
    report.n_events = int(lab.max())
    report.contacts = event_contacts(lab, group, eu, ev, n)
    report.lineage = _relabel_lineage(lineage, u)
    if calibration is not None:
        calibration._cache, calibration._cache_key = None, None
    return lab, report


@dataclass
class CoactivationReport:
    """What the detector chose for a movie.

    Attributes:
        level_quantile (float): co-activation level of the events, as a quantile of the null.
        level (float): the same level in dR units.
        onset_window (int): onset window delta (frames).
        core (int): density core m.
        min_event_size (int): minimum number of pulses of an event.
        seed_size (int): minimum co-active patch that can start a new event.
        prune_window (int | None): k for removing early members (None: not applied).
        baseline_window (int): window of the local baseline (frames).
        smooth_k (int): temporal smoothing window of the signal (frames).
        n_events (int): number of events.
        table (pd.DataFrame): every level and core tried.
        contacts (pd.DataFrame): pairs of events that touch.
        lineage (pd.DataFrame | None): merges (allow_merges=True).
    """

    level_quantile: float = float("nan")
    level: float = float("nan")
    onset_window: int = 0
    core: int = 0
    min_event_size: int = 0
    seed_size: int = 0
    prune_window: int | None = None
    baseline_window: int = 0
    smooth_k: int = 3
    n_events: int = 0
    table: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    contacts: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    lineage: pd.DataFrame | None = field(default=None, repr=False)
    calibration_data: dict | None = field(default=None, repr=False)


# ============================================================================ public API (tracked cells)
def _check_epsilon(epsilon):
    if not epsilon > 0:
        raise ValueError(f"epsilon must be > 0 (expected false events per movie), got {epsilon}")


def _to_matrix(df, position_columns, frame_column, obj_id_column, measurement_column):
    if df.duplicated([frame_column, obj_id_column]).any():
        raise ValueError(f"Duplicate rows for the same {frame_column} and {obj_id_column}")
    frames = np.sort(df[frame_column].unique())
    if np.issubdtype(frames.dtype, np.integer) and frames.size > 1:
        # keep frames missing for every unit as empty rows, so that lags stay in frames (step: the common spacing)
        step = int(np.gcd.reduce(np.diff(frames)))
        frames = np.arange(frames[0], frames[-1] + 1, step, dtype=frames.dtype)
    ids = np.sort(df[obj_id_column].unique())
    rows = (np.searchsorted(frames, df[frame_column].to_numpy()), np.searchsorted(ids, df[obj_id_column].to_numpy()))
    T, n = frames.size, ids.size
    observed = np.zeros((T, n), bool)
    observed[rows] = True
    S = np.full((T, n), np.nan)
    S[rows] = df[measurement_column].to_numpy(float)
    pos_t = np.full((T, n, len(position_columns)), np.nan)
    pos_t[rows] = df[position_columns].to_numpy(float)
    return S, observed, pos_t, rows


def detect_coactivation_events(
    X: pd.DataFrame,
    position_columns: list,
    frame_column: str,
    obj_id_column: str,
    measurement_column: str,
    epsilon: float = 1.0,
    clid_column: str = "collid",
    remove_population_trend: bool = True,
    baseline_window: int | None = None,
    allow_merges: bool = False,
    refine_start: bool = True,
    prune_early: bool = True,
    calibration: CoactivationCalibration | None = None,
    n_null: int = 8,
    seed: int = 0,
    return_report: bool = False,
    smooth_k: int = 3,
):
    """Detect collective events in tracked single-cell time series, directly from the continuous signal.

    No binarisation and no distance or size parameters: neighbours come from a Delaunay graph per frame, and
    every threshold is calibrated against copies of the data in which each cell's trace is shifted in time.

    Arguments:
        X (pd.DataFrame): long table, one row per cell and frame.
        position_columns (list): names of the coordinate columns.
        frame_column (str): name of the frame column.
        obj_id_column (str): name of the track id column.
        measurement_column (str): name of the (continuous) signal column.
        epsilon (float): number of false events one is willing to expect per movie. Default 1.
        clid_column (str): name of the output column with the event ids. Default "collid".
        remove_population_trend (bool): remove drift shared by all cells (bleaching, focus); a pulse of the whole
            field is kept. Default True.
        baseline_window (int | None): window (frames) of each cell's own baseline; None chooses it from the data.
        allow_merges (bool): two waves that started separately, out of one onset group, continue as one event
            from their first contact (lineage in the report). Default False.
        refine_start (bool): read each member's first frame on the unsmoothed signal. Default True.
        prune_early (bool): remove members that started well before every neighbouring member. Default True.
        calibration (CoactivationCalibration | None): frozen definition from calibrate_coactivation.
        n_null (int): number of time-shifted null copies. Default 8.
        seed (int): random seed of the null copies.
        return_report (bool): also return a CoactivationReport.
        smooth_k (int): odd number of frames of the centred running mean applied to each trace before
            everything else (1 = no smoothing). Larger values suppress more frame-to-frame noise but blur onsets;
            refine_start reads the starts back on the unsmoothed signal. Default 3, as in binData.

    Returns:
        pd.DataFrame: the rows of X that belong to an event, with the event id in clid_column
        (and the report if return_report).
    """
    if X.empty:
        raise ValueError("Input is empty")
    S, observed, pos_t, rows = _to_matrix(X, position_columns, frame_column, obj_id_column, measurement_column)
    graph = CellGraph(pos_t, observed)
    lab, report = detect_matrix(
        S,
        observed,
        graph,
        epsilon=epsilon,
        n_null=n_null,
        seed=seed,
        remove_population_trend=remove_population_trend,
        baseline_window=baseline_window,
        allow_merges=allow_merges,
        refine_start=refine_start,
        prune_early=prune_early,
        calibration=calibration,
        smooth_k=smooth_k,
    )
    res = X.copy()
    res[clid_column] = lab[rows].astype(np.int64)
    res = res[res[clid_column] > 0].reset_index(drop=True)
    return (res, report) if return_report else res


def calibrate_coactivation(
    movies: list,
    position_columns: list,
    frame_column: str,
    obj_id_column: str,
    measurement_column: str,
    epsilon: float = 1.0,
    n_null: int = 8,
    seed: int = 0,
    level_mode: str = "quantile",
    remove_population_trend: bool = True,
    smooth_k: int = 3,
):
    """One event definition for several movies (replicates, conditions, or a pilot for online use).

    Pass 1 runs the null pipeline on every movie and pools the neighbour-lag counts to fix one onset window per
    level. Pass 2 re-runs the group-size nulls at that window and pools them, so thresholds are better resolved
    than from one movie. The level and core with the most significant events over all movies are kept.

    Arguments:
        movies (list): DataFrames with the same columns.
        level_mode (str): "quantile" (each movie gets its level from its own null quantile; robust to exposure
            changes) or "absolute" (the calibration's levels in signal units).
        smooth_k (int): temporal smoothing window, as in detect_coactivation_events (stored in the calibration).

    Returns:
        CoactivationCalibration: pass it to detect_coactivation_events(..., calibration=...).
    """
    mats = []
    for df in movies:
        S, observed, pos_t, _ = _to_matrix(df, position_columns, frame_column, obj_id_column, measurement_column)
        mats.append((S, observed, CellGraph(pos_t, observed)))
    return _calibrate(mats, epsilon, n_null, seed, level_mode, remove_population_trend, smooth_k)


def _calibrate(mats, epsilon, n_null, seed, level_mode, remove_population_trend, smooth_k=3):
    if not mats:
        raise ValueError("calibrate_coactivation needs at least one movie")
    kw = dict(
        epsilon=epsilon,
        n_null=n_null,
        remove_population_trend=remove_population_trend,
        return_calibration_data=True,
        smooth_k=smooth_k,
    )
    lag = {}
    for i, (S, observed, graph) in enumerate(mats):
        _, rep = detect_matrix(S, observed, graph, seed=seed + i, **kw)
        for q, d in rep.calibration_data["levels"].items():
            r = lag.setdefault(q, [np.zeros_like(d["lag_real"]), []])
            r[0] = r[0] + d["lag_real"]
            r[1].append(d["lag_null"])
    deltas, strict = {}, {}
    for q, (real, nls) in lag.items():
        nl = np.sum(nls, axis=0)
        mu = nl.mean(0)
        z = (real - mu) / (nl.std(0) + np.sqrt(mu + 1))
        ok = (real >= 2 * mu) & (z > 3)
        deltas[q] = int(np.flatnonzero(ok).max()) if ok.any() else 0
        strict[q] = _strict_window(real, mu, z, STRICT_RATIO, deltas[q])
    if not deltas:
        raise ValueError("No co-active pulses in any calibration movie")
    pooled = {(q, m): [] for q in deltas for m in CORE_LADDER}
    real = {(q, m): [] for q in deltas for m in CORE_LADDER}  # (sizes, unit-frames) per movie with pulses at q
    levels = {q: [] for q in deltas}
    null_cf = {q: 0.0 for q in deltas}  # unit-frames behind the pooled null sizes, per level
    for i, (S, observed, graph) in enumerate(mats):
        _, rep = detect_matrix(S, observed, graph, seed=seed + 100 + i, fixed_deltas=deltas, **kw)
        cd = rep.calibration_data
        for q, d in cd["levels"].items():
            levels[q].append(d["h"])
            null_cf[q] += cd["n_obs"] * cd["n_null"]
            for m in CORE_LADDER:
                pooled[(q, m)].append(d["null_sizes"][m])
                real[(q, m)].append((d["sizes"][m], cd["n_obs"]))
    # levels that found no pulses in the second pass cannot be calibrated
    for q in [q for q, v in levels.items() if not v]:
        for d_ in (deltas, strict, levels, null_cf):
            del d_[q]
        for m in CORE_LADDER:
            del pooled[(q, m)], real[(q, m)]
    if not deltas:
        raise ValueError("No co-active pulses in any calibration movie")
    pooled = {k: np.concatenate(v) if v else np.zeros(0, int) for k, v in pooled.items()}
    calib = CoactivationCalibration(
        q=0,
        q_lo=min(deltas),
        m=0,
        levels={q: float(np.median(v)) for q, v in levels.items()},
        deltas=deltas,
        strict_deltas=strict,
        null_sizes=pooled,
        null_cellframes=null_cf,
        epsilon=epsilon,
        level_mode=level_mode,
        smooth_k=int(smooth_k),
    )
    n_tests = len(deltas) * len(CORE_LADDER)
    best = None
    for (q, m), items in real.items():
        n_ev = sum(int((sz >= calib.threshold(q, m, N, n_tests)).sum()) for sz, N in items)
        if best is None or (n_ev, q, m) > best[0]:
            best = ((n_ev, q, m), q, m)
    assert best is not None  # deltas is not empty
    calib.q, calib.m = best[1], best[2]
    return calib
