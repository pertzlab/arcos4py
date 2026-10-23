"""Numba kernels for co-activation event detection.

All kernels are deterministic and reproduce the reference (pandas/numpy) implementation exactly;
random draws stay in Python so that results do not depend on whether numba is used.
"""

from __future__ import annotations

import numpy as np
from numba import njit


# ----------------------------------------------------------------------------- null copies
@njit(cache=True)
def roll_within_lifetime(D, observed, offsets, fill):
    """Shift each unit's values along its observed frames by its own offset (np.roll within lifetime).

    Arguments:
        D (np.ndarray): (T, n) values.
        observed (np.ndarray): (T, n) bool, frames where the unit is present.
        offsets (np.ndarray): (n,) int, roll offset per unit.
        fill (float): value outside the observed frames.

    Returns:
        np.ndarray: (T, n) shifted values.
    """
    T, n = D.shape
    out = np.full((T, n), fill, dtype=D.dtype)
    idx = np.empty(T, np.int64)
    for j in range(n):
        m = 0
        for t in range(T):
            if observed[t, j]:
                idx[m] = t
                m += 1
        if m == 0:
            continue
        k = offsets[j] % m
        for i in range(m):
            out[idx[(i + k) % m], j] = D[idx[i], j]
    return out


# ----------------------------------------------------------------------------- pulses
@njit(cache=True)
def pulse_timing(cells, onsets, offsets, R):
    """Pulse time = frame of the steepest rise from 2 frames before the call to the middle of the call."""
    T = R.shape[0]
    out = np.empty(cells.size, np.int64)
    for k in range(cells.size):
        a = onsets[k]
        lo = max(a - 2, 0)
        hi = min(a + max((offsets[k] - a) // 2, 1) + 1, T)
        best = lo
        bv = R[lo, cells[k]]
        for t in range(lo + 1, hi):
            v = R[t, cells[k]]
            if v > bv:
                bv = v
                best = t
        out[k] = best
    return out


@njit(cache=True)
def _first_at_least(arr, lo, hi, value):
    """First index in arr[lo:hi] (sorted ascending) with arr[index] >= value (hi if none)."""
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _pulse_pairs(cells, times, cell_ptr, sorted_times, sorted_pid, indptr, indices, max_lag, mode, rows, cols, hist):
    """Shared traversal over linked pulse pairs.

    Pulse p, each graph neighbour j of its cell in the frame of its time, every pulse q of j with time in
    [t_p, t_p + max_lag]. mode 0: count pairs; 1: fill rows/cols; 2: histogram of lags.
    """
    n_frames = indptr.shape[0]
    n_pairs = 0
    for p in range(cells.size):
        c = cells[p]
        t0 = times[p]
        f = t0 if n_frames > 1 else 0
        if f < 0 or f >= n_frames:
            continue
        for e in range(indptr[f, c], indptr[f, c + 1]):
            j = indices[e]
            q = _first_at_least(sorted_times, cell_ptr[j], cell_ptr[j + 1], t0)
            while q < cell_ptr[j + 1] and sorted_times[q] - t0 <= max_lag:
                if mode == 1:
                    rows[n_pairs] = p
                    cols[n_pairs] = sorted_pid[q]
                elif mode == 2:
                    hist[sorted_times[q] - t0] += 1
                n_pairs += 1
                q += 1
    return n_pairs


def _by_cell(cells, times, n_cells):
    order = np.lexsort((times, cells))
    cell_ptr = np.zeros(n_cells + 1, np.int64)
    np.cumsum(np.bincount(cells, minlength=n_cells), out=cell_ptr[1:])
    return cell_ptr, np.ascontiguousarray(times[order], np.int64), np.ascontiguousarray(order, np.int64)


def pulse_pairs(cells, times, n_cells, indptr, indices, max_lag):
    """Linked pulse pairs (p, q): q in a graph neighbour of p's cell (graph of p's frame), t_p <= t_q <= t_p + max_lag."""
    cells = np.ascontiguousarray(cells, np.int64)
    times = np.ascontiguousarray(times, np.int64)
    cell_ptr, st, sp = _by_cell(cells, times, n_cells)
    empty = np.empty(0, np.int64)
    n_pairs = _pulse_pairs(cells, times, cell_ptr, st, sp, indptr, indices, max_lag, 0, empty, empty, empty)
    rows, cols = np.empty(n_pairs, np.int64), np.empty(n_pairs, np.int64)
    _pulse_pairs(cells, times, cell_ptr, st, sp, indptr, indices, max_lag, 1, rows, cols, empty)
    return rows, cols


def lag_histogram(cells, times, n_cells, indptr, indices, max_lag):
    """Number of neighbour pulse pairs per onset lag 0..max_lag."""
    cells = np.ascontiguousarray(cells, np.int64)
    times = np.ascontiguousarray(times, np.int64)
    cell_ptr, st, sp = _by_cell(cells, times, n_cells)
    hist = np.zeros(max_lag + 1, np.int64)
    empty = np.empty(0, np.int64)
    _pulse_pairs(cells, times, cell_ptr, st, sp, indptr, indices, max_lag, 2, empty, empty, hist)
    return hist


# ----------------------------------------------------------------------------- membership
@njit(cache=True)
def halfmax_extend(out, lab, Dz, observed, th):
    """Run extension of half-maximum membership (in place on `out`).

    A member keeps its event id forwards and backwards in time while its dR stays above the event's threshold.
    """
    T, n = out.shape
    for t in range(1, T):
        for c in range(n):
            p = out[t - 1, c]
            if p > 0 and out[t, c] == 0 and observed[t, c] and Dz[t, c] > th[p]:
                out[t, c] = p
    for t in range(T - 2, -1, -1):
        for c in range(n):
            p = out[t + 1, c]
            if p > 0 and out[t, c] == 0 and observed[t, c] and Dz[t, c] > th[p]:
                out[t, c] = p
    return out


@njit(cache=True)
def refine_starts(out, item_e, item_c, item_s0, item_th, D_raw, observed, reach):
    """Start of each (event, cell) membership read on the unsmoothed dR (see raw_start), in place on `out`.

    Items are processed in (event, cell) order; each moves only the start of the cell's first run.
    """
    T = out.shape[0]
    for k in range(item_e.size):
        e, c, s0, th = item_e[k], item_c[k], item_s0[k], item_th[k]
        f = s0
        lo = max(s0 - reach, 0)
        while f - 1 >= lo and observed[f - 1, c] and out[f - 1, c] == 0 and D_raw[f - 1, c] >= th:
            f -= 1
        if f < s0:
            for t in range(f, s0):
                out[t, c] = e
        else:
            r1 = s0
            while r1 + 1 < T and out[r1 + 1, c] == e:
                r1 += 1
            hi = min(s0 + reach, r1)
            while f < hi and D_raw[f, c] < th:
                f += 1
            for t in range(s0, f):
                out[t, c] = 0
    return out


@njit(cache=True)
def early_outliers(item_e, item_c, item_s0, start_of, indptr, indices, k):
    """For each (event, cell) item: True if the cell is an early outlier of its event.

    An outlier started more than k frames before every neighbouring member of the same event (neighbours from
    the graph of its start frame), or has no neighbouring member.

    start_of: (n_events + 1, n) start frame of each event's members (-1 if not a member).
    """
    n_frames = indptr.shape[0]
    bad = np.zeros(item_e.size, np.bool_)
    for i in range(item_e.size):
        e, c, t = item_e[i], item_c[i], item_s0[i]
        f = t if n_frames > 1 else 0
        keep = False
        for q in range(indptr[f, c], indptr[f, c + 1]):
            tj = start_of[e, indices[q]]
            if tj >= 0 and tj - t <= k:  # a neighbour started at most k frames after (or before) it
                keep = True
                break
        bad[i] = not keep
    return bad


# ----------------------------------------------------------------------------- baseline
@njit(cache=True)
def rolling_median_centered(X, W):
    """Centred running median over W frames per column, NaNs ignored, NaN where a window has no values.

    Same windows as pandas .rolling(W, center=True, min_periods=1).median(): the window of frame i covers
    frames [i + (W - 1) // 2 - W + 1, i + (W - 1) // 2], clipped to the movie.
    """
    T, n = X.shape
    out = np.full((T, n), np.nan)
    off = (W - 1) // 2
    buf = np.empty(W, np.float64)
    for j in range(n):
        m = 0  # sorted non-NaN values of the current window
        lo, hi = 0, 0  # window [lo, hi)
        for i in range(T):
            new_hi = min(i + off + 1, T)
            new_lo = max(i + off + 1 - W, 0)
            while hi < new_hi:  # add X[hi]
                v = X[hi, j]
                if v == v:
                    k = m
                    while k > 0 and buf[k - 1] > v:
                        buf[k] = buf[k - 1]
                        k -= 1
                    buf[k] = v
                    m += 1
                hi += 1
            while lo < new_lo:  # remove X[lo]
                v = X[lo, j]
                if v == v:
                    k = 0
                    while buf[k] != v:
                        k += 1
                    for q in range(k, m - 1):
                        buf[q] = buf[q + 1]
                    m -= 1
                lo += 1
            if m > 0:
                if m % 2 == 1:
                    out[i, j] = buf[m // 2]
                else:
                    out[i, j] = (buf[m // 2 - 1] + buf[m // 2]) / 2
    return out


# ----------------------------------------------------------------------------- frame clusters
@njit(cache=True)
def _find(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


@njit(cache=True)
def frame_cluster_sizes(active, indptr, indices):
    """Sizes of the connected clusters of active units in every frame (graph of that frame).

    All frames are concatenated; the order of sizes is arbitrary.
    """
    T, n = active.shape
    n_frames = indptr.shape[0]
    parent = np.arange(n)
    size = np.zeros(n, np.int64)
    out = np.empty(active.sum(), np.int64)
    n_out = 0
    for t in range(T):
        f = t if n_frames > 1 else 0
        for c in range(n):
            parent[c] = c
        for c in range(n):
            if not active[t, c]:
                continue
            for e in range(indptr[f, c], indptr[f, c + 1]):
                j = indices[e]
                if active[t, j]:
                    a, b = _find(parent, c), _find(parent, j)
                    if a != b:
                        parent[max(a, b)] = min(a, b)
        for c in range(n):
            size[c] = 0
        for c in range(n):
            if active[t, c]:
                size[_find(parent, c)] += 1
        for c in range(n):
            if size[c] > 0:
                out[n_out] = size[c]
                n_out += 1
    return out[:n_out]


# ----------------------------------------------------------------------------- births (frame-by-frame tracking)
@njit(cache=True)
def _clusters(act, na, active_row, f, indptr, indices, parent, rootcl):
    """Connected clusters of the active units act[:na] in frame f, numbered by their smallest unit index."""
    for i in range(na):
        parent[act[i]] = act[i]
    for i in range(na):
        c = act[i]
        for e in range(indptr[f, c], indptr[f, c + 1]):
            j = indices[e]
            if active_row[j]:
                a = _find(parent, c)
                b = _find(parent, j)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    cl = np.empty(na, np.int64)
    for i in range(na):
        rootcl[_find(parent, act[i])] = -1
    nc = 0
    for i in range(na):
        r = _find(parent, act[i])
        if rootcl[r] < 0:
            rootcl[r] = nc
            nc += 1
        cl[i] = rootcl[r]
    return cl, nc


@njit(cache=True)
def _within_hops(valid, f, indptr, indices, hops, depth, queue):
    """depth[u] = hops from the nearest valid unit (-1 if more than `hops`); returns whether any unit is valid."""
    nq = 0
    for c in range(valid.size):
        depth[c] = -1
        if valid[c]:
            depth[c] = 0
            queue[nq] = c
            nq += 1
    head = 0
    while head < nq:
        c = queue[head]
        head += 1
        if depth[c] >= hops:
            continue
        for e in range(indptr[f, c], indptr[f, c + 1]):
            j = indices[e]
            if depth[j] < 0:
                depth[j] = depth[c] + 1
                queue[nq] = j
                nq += 1
    return nq > 0


@njit(cache=True)
def _majority(ring, nr, valid, mem_lab, counts):
    """Most common memory label among the valid units of the ring (ties: smallest); 0 if none is valid."""
    for q in range(nr):
        u = ring[q]
        if valid[u]:
            counts[mem_lab[u]] += 1
    best_l, best_c = 0, 0
    for q in range(nr):
        u = ring[q]
        if valid[u]:
            lv = mem_lab[u]
            if counts[lv] > best_c or (counts[lv] == best_c and lv < best_l):
                best_c, best_l = counts[lv], lv
    for q in range(nr):
        if valid[ring[q]]:
            counts[mem_lab[ring[q]]] = 0
    return best_l


@njit(cache=True)
def _nearest_label(src, valid, mem_lab, f, indptr, indices, seen, vis, ring, nxt, counts):
    """Label of the first ring around src (src itself first) that holds valid memory units, up to 64 rings."""
    seen[src] = 1
    vis[0], ring[0] = src, src
    nv, nr, label = 1, 1, 0
    for _ in range(64):
        label = _majority(ring, nr, valid, mem_lab, counts)
        if label > 0:
            break
        nn = 0
        for q in range(nr):
            u = ring[q]
            for e in range(indptr[f, u], indptr[f, u + 1]):
                j = indices[e]
                if seen[j] < 0:
                    seen[j] = 1
                    vis[nv] = j
                    nv += 1
                    nxt[nn] = j
                    nn += 1
        if nn == 0:
            break
        ring[:nn] = nxt[:nn]
        nr = nn
    for q in range(nv):
        seen[vis[q]] = -1
    return label


@njit(cache=True)
def track_seeded(mask, seedable, observed, indptr, indices, seed_size, n_prev, link_hops):
    """ARCOS-style tracking of one event's unit-frames (mask), frame by frame.

    Per frame: clusters = connected active units (graph of that frame). A cluster is linked if any of its units
    lies within link_hops hops of a unit labelled in the last n_prev frames; each unit of a linked cluster takes
    the most common label among the labelled units of the first ring around it that holds any (ties: smallest
    label). An unlinked cluster starts a new label only if it holds >= seed_size seedable unit-frames; new
    labels follow the clusters' smallest unit index.
    """
    T, n = mask.shape
    n_frames = indptr.shape[0]
    out = np.zeros((T, n), np.int32)
    mem_lab = np.zeros(n, np.int32)
    mem_t = np.full(n, -(10**9), np.int64)
    parent, rootcl = np.arange(n), np.full(n, -1, np.int64)
    depth, queue = np.full(n, -1, np.int64), np.empty(n, np.int64)
    seen, vis = np.full(n, -1, np.int64), np.empty(n, np.int64)
    ring, nxt = np.empty(n, np.int64), np.empty(n, np.int64)
    act = np.empty(n, np.int64)
    valid = np.zeros(n, np.bool_)
    next_id = 1
    for t in range(T):
        f = t if n_frames > 1 else 0
        na = 0
        for c in range(n):
            if mask[t, c]:
                act[na] = c
                na += 1
        if na == 0:
            continue
        cl, nc = _clusters(act, na, mask[t], f, indptr, indices, parent, rootcl)
        for c in range(n):
            valid[c] = mem_lab[c] > 0 and mem_t[c] >= t - n_prev and observed[t, c]
        counts = np.zeros(next_id + 1, np.int64)
        lab = np.zeros(na, np.int32)
        linked = np.zeros(na, np.bool_)
        if _within_hops(valid, f, indptr, indices, link_hops, depth, queue):
            cl_linked = np.zeros(nc, np.bool_)
            for i in range(na):
                if depth[act[i]] >= 0:
                    cl_linked[cl[i]] = True
            for i in range(na):
                linked[i] = cl_linked[cl[i]]
                if linked[i]:
                    lab[i] = _nearest_label(act[i], valid, mem_lab, f, indptr, indices, seen, vis, ring, nxt, counts)
        nseed = np.zeros(nc, np.int64)
        for i in range(na):
            if not linked[i] and seedable[t, act[i]]:
                nseed[cl[i]] += 1
        newid = np.full(nc, -1, np.int64)
        for k in range(nc):
            if nseed[k] >= max(seed_size, 1):
                newid[k] = next_id
                next_id += 1
        for i in range(na):
            if not linked[i] and newid[cl[i]] > 0:
                lab[i] = newid[cl[i]]
            if lab[i] > 0:
                out[t, act[i]] = lab[i]
                mem_lab[act[i]] = lab[i]
                mem_t[act[i]] = t
    return out
