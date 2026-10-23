"""Activity (dR) with a baseline estimated only from frames where a unit is at rest.

Model: X[t, i] = a_i + b_i * P(t) + u_i(t) + activity + noise, with X the 3-frame mean of the signal.

- P(t): drift shared by all units (bleaching, focus, lamp), each unit with its own offset a_i and scale b_i,
  estimated per frame from the units resting in that frame only. Where (almost) no unit rests, e.g. during
  a pulse of the whole field, P is interpolated and the pulse stays in the data.
- u_i(t): what is left of the unit's own slow drift, a robust local line through its resting frames in a
  window of W frames (no lag at the start and end of the movie), interpolated through its activations.

Resting frames come from the unit's own trace relative to its own noise (sigma = MAD): runs beyond +-1 sigma
that reach +-3 sigma are excluded (both signs), padded by 2 frames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from ._coactivation_kernels import rolling_median_centered


def pulses_from_binary(B):
    """Runs of True per unit: arrays (unit, onset, offset_exclusive), sorted by unit then onset."""
    T, n = B.shape
    d = np.diff(np.vstack([np.zeros((1, n), np.int8), B.astype(np.int8), np.zeros((1, n), np.int8)]), axis=0)
    on_t, on_c = np.nonzero(d == 1)
    off_t, off_c = np.nonzero(d == -1)
    o1 = np.lexsort((on_t, on_c))
    o2 = np.lexsort((off_t, off_c))
    return on_c[o1], on_t[o1], off_t[o2]


def interp_columns(X, inside_only):
    """Linear interpolation of NaNs along time, column by column, exactly as pandas .interpolate().

    np.interp on frame positions: inside_only fills only gaps between the first and last value of a column
    (limit_area="inside"); otherwise every NaN is filled, holding the edge values (limit_direction="both").
    """
    X = np.array(X, dtype=np.float64, copy=True)
    nan = np.isnan(X)
    idx = np.arange(X.shape[0])
    for j in np.flatnonzero(nan.any(0) & (~nan).any(0)):
        valid = ~nan[:, j]
        vi = idx[valid]
        fill = nan[:, j] & (idx > vi[0]) & (idx < vi[-1]) if inside_only else nan[:, j]
        if fill.any():
            X[fill, j] = np.interp(idx[fill], vi, X[valid, j])
    return X


def smoothed(S, observed, smooth=3):
    """Signal interpolated inside each unit's lifetime and averaged over `smooth` frames (NaN outside)."""
    T = S.shape[0]
    any_obs = observed.any(0)
    first = np.where(any_obs, observed.argmax(0), T)
    last = np.where(any_obs, T - 1 - observed[::-1].argmax(0), -1)
    tt = np.arange(T)[:, None]
    life = (tt >= first[None]) & (tt <= last[None])
    X = interp_columns(np.where(observed, S, np.nan), inside_only=True)
    X = np.where(life, X, np.nan)
    if smooth and smooth > 1:
        X = pd.DataFrame(X).rolling(smooth, center=True, min_periods=1).mean().to_numpy()
    return np.where(life, X, np.nan)


def _hysteresis_runs(Z, life, lo, hi):
    """Runs along time (per unit) where Z > lo that contain at least one frame with Z > hi."""
    st = np.zeros((3, 3), bool)
    st[:, 1] = True
    lab, k = ndi.label(life & (Z > lo), structure=st)
    if k == 0:
        return np.zeros_like(life)
    keep = np.zeros(k + 1, bool)
    keep[np.unique(lab[life & (Z > hi)])] = True
    keep[0] = False
    return keep[lab]


def _symmetric_mask(D, life, z_hi, z_lo, resting=None):
    """Mask of runs beyond +-z_lo sigma that reach +-z_hi sigma (both signs), padded by 2 frames."""
    Dr = np.where(life & (True if resting is None else resting), D, np.nan)
    med = np.nanmedian(Dr, axis=0)
    sig = 1.4826 * np.nanmedian(np.abs(Dr - med[None, :]), axis=0)
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.inf)
    Z = np.where(life, (D - np.nan_to_num(med)[None, :]) / sig[None, :], 0.0)
    Mpos = _hysteresis_runs(Z, life, z_lo, z_hi)
    Mneg = _hysteresis_runs(-Z, life, z_lo, z_hi)
    pad = np.ones((5, 1), bool)
    M = (ndi.binary_dilation(Mpos, structure=pad) | ndi.binary_dilation(Mneg, structure=pad)) & life
    return M, Mpos, Z


def _population_drift(X, use, observed, min_rest=0.05, n_inner=3):
    """Robust rank-1 drift X[t, i] ~ a_i + b_i * P(t), fitted on resting unit-frames only."""
    T, n = X.shape
    a = np.nanmedian(np.where(use, X, np.nan), axis=0)
    a = np.where(np.isfinite(a), a, np.nanmedian(X, axis=0))
    bcoef = np.ones(n)
    n_present = np.maximum(observed.sum(1), 1)
    P = np.zeros(T)
    cnt = np.zeros(T, int)
    for _ in range(n_inner):
        ok_b = np.abs(bcoef) > 0.2 * np.median(np.abs(bcoef))
        sc = np.where(ok_b, bcoef, 1.0)[None, :]
        V = np.where(use & ok_b[None, :], (X - a[None, :]) / sc, np.nan)
        cnt = np.isfinite(V).sum(1)
        with np.errstate(all="ignore"):
            P = np.nanmedian(V, axis=1)
        P[cnt < np.maximum(3, min_rest * n_present)] = np.nan
        if np.all(np.isnan(P)):
            return np.zeros_like(X), np.zeros(T), np.ones(T, bool)
        P = pd.Series(P).interpolate(limit_direction="both").to_numpy()
        P = P - np.median(P)
        w = use.astype(float)
        nw = np.maximum(w.sum(0), 1)
        Pm = (w * P[:, None]).sum(0) / nw
        Xm = np.nansum(np.where(use, X, 0.0), axis=0) / nw
        cov = np.nansum(np.where(use, (X - Xm) * (P[:, None] - Pm), 0.0), axis=0)
        var = (w * (P[:, None] - Pm) ** 2).sum(0)
        bcoef = np.where(var > 1e-12, cov / np.maximum(var, 1e-12), 1.0)
        a = Xm - bcoef * Pm
    filled = cnt < np.maximum(3, min_rest * n_present)
    return a[None, :] + bcoef[None, :] * P[:, None], P, filled


def _rolling_line(R, use, W, min_frac=0.2):
    """Per unit, robust local line through the frames in `use` in a centred window of W frames.

    Level from their running median, slope from least squares (removes the lag of a plain running median at
    the movie edges and next to long activations). Windows with too few usable frames are interpolated.
    """
    T, n = R.shape
    w = use.astype(float)
    r = np.where(use, R, 0.0)
    tt = (np.arange(T, dtype=float) - T / 2)[:, None]

    def csum(a):
        return np.vstack([np.zeros((1, n)), np.cumsum(a, axis=0)])

    C0, C1, C2, Cy, Cty = csum(w), csum(w * tt), csum(w * tt * tt), csum(r), csum(r * tt)
    h = W // 2
    lo = np.clip(np.arange(T) - h, 0, T)
    hi = np.clip(np.arange(T) + h + 1, 0, T)
    S0, S1, S2, Sy, Sty = (C[hi] - C[lo] for C in (C0, C1, C2, Cy, Cty))
    den = S0 * S2 - S1**2
    with np.errstate(all="ignore"):
        slope = np.where(den > 1e-9, (S0 * Sty - S1 * Sy) / den, 0.0)
    med = rolling_median_centered(np.ascontiguousarray(np.where(use, R, np.nan), np.float64), int(W))
    with np.errstate(all="ignore"):
        base = med + slope * (tt - S1 / S0)
    base[S0 < max(5, min_frac * W)] = np.nan
    return np.nan_to_num(interp_columns(base, inside_only=False), nan=0.0)


def masked_baseline_activity(S, observed, population=True, window=None, window_mult=5, n_iter=3, z_hi=3.0, z_lo=1.0):
    """Activity dR (T, n) with a resting-frame baseline (see module docstring).

    Arguments:
        S (np.ndarray): (T, n) signal, NaN where a unit is absent.
        observed (np.ndarray): (T, n) bool.
        population (bool): remove the drift shared by all units.
        window (int | None): window of the local baseline in frames; None = window_mult x the 95th
            percentile of activation lengths (at least 51 frames, at most the movie).

    Returns:
        tuple: dR (NaN outside lifetimes) and a dict with the baseline window and masked fraction.
    """
    T, n = observed.shape
    X = smoothed(S, observed, 3)
    life = np.isfinite(X)
    W0 = int(min(T, max(51, T // 4)))
    base0 = _rolling_line(X, life, W0)
    M, Mpos, Z = _symmetric_mask(np.where(life, X - base0, np.nan), life, z_hi, z_lo)
    W = T
    D = np.where(life, X, np.nan)
    for _ in range(n_iter):
        use = life & ~M
        if population:
            shared, _, _ = _population_drift(X, use & (np.abs(Z) <= z_lo), observed)
            R = X - shared
        else:
            R = X - np.nanmedian(np.where(use, X, np.nan), axis=0)[None, :]
        _, a_, b_ = pulses_from_binary(Mpos)
        lens = (b_ - a_) if a_.size else np.array([1])
        W = int(min(T, max(51, window_mult * np.quantile(lens, 0.95)))) if window is None else int(min(T, window))
        base = _rolling_line(R, use, W)
        D = np.where(life, R - base, np.nan)
        M, Mpos, Z = _symmetric_mask(D, life, z_hi, z_lo, resting=~M)
    return D, dict(window=W, masked_fraction=float(M[observed].mean()) if observed.any() else 0.0)
