"""Co-activation event detection for image stacks (T, Y, X), through square tiles on a fixed grid.

Each tile is one unit: its mean signal is the trace, its neighbours are the 8 surrounding tiles, and everything
else (baseline, co-activation score, time-shift null, onset groups, births) is the cell-level detector.
"""

from __future__ import annotations

import numpy as np

from ._coactivation import CoactivationCalibration, detect_matrix
from ._coactivation_graph import GridGraph


def bin_stack(stack, b):
    """Mean over b x b tiles (edges that do not fill a tile are dropped)."""
    T, Y, X = stack.shape
    return stack[:, : Y // b * b, : X // b * b].reshape(T, Y // b, b, X // b, b).astype(np.float32).mean((2, 4))


def noise_scale(stack, max_d=12, frames=60, level=0.2):
    """Distance (px) beyond which frame-to-frame changes are no longer correlated between pixels: the optical
    blur for subcellular data, the object size when objects move as a whole."""
    T = stack.shape[0]
    idx = np.linspace(0, T - 2, min(T - 1, frames)).astype(int)
    d = stack[idx + 1].astype(np.float32) - stack[idx].astype(np.float32)
    d -= d.mean(axis=(1, 2), keepdims=True)

    def corr(k):
        a, b = d[:, :, :-k].ravel(), d[:, :, k:].ravel()
        c, e = d[:, :-k, :].ravel(), d[:, k:, :].ravel()
        return 0.5 * (np.corrcoef(a, b)[0, 1] + np.corrcoef(c, e)[0, 1])

    for k in range(1, max_d + 1):
        if corr(k) < level:
            return k
    return max_d


def auto_mask(stack):
    """Foreground by Otsu on the time-averaged image, only if it clearly stands out (at least twice as bright
    as the background); otherwise everything."""
    from skimage.filters import threshold_otsu

    m = stack.mean(0)
    fg = m > threshold_otsu(m)
    if fg.any() and (~fg).any() and m[fg].mean() > 2 * max(m[~fg].mean(), 1e-9):
        return fg
    return np.ones(m.shape, bool)


def _tiles_once(stack, tile, mask, **kw):
    T, Y, X = stack.shape
    B = bin_stack(stack, tile)
    by, bx = B.shape[1:]
    frac = bin_stack(mask[None].astype(np.float32), tile)[0]
    keep = np.argwhere(frac >= 0.5)
    tid = keep[:, 0] * bx + keep[:, 1]
    S = B.reshape(T, -1)[:, tid].astype(np.float64)
    observed = np.ones(S.shape, bool)
    labt, report = detect_matrix(S, observed, GridGraph(keep, T), **kw)
    full = np.zeros((T, by * bx), np.int32)
    full[:, tid] = labt
    up = np.repeat(np.repeat(full.reshape(T, by, bx), tile, 1), tile, 2)
    lab = np.zeros((T, Y, X), np.int32)
    lab[:, : up.shape[1], : up.shape[2]] = up
    lab *= mask[None]
    report.tile = tile
    return lab, report


def detect_coactivation_events_image(
    stack: np.ndarray,
    epsilon: float = 1.0,
    tile: int | str = "auto",
    mask: np.ndarray | str | None = "auto",
    remove_population_trend: bool = True,
    baseline_window: int | None = None,
    allow_merges: bool = False,
    refine_start: bool = True,
    prune_early: bool = True,
    calibration: CoactivationCalibration | None = None,
    n_null: int = 8,
    seed: int = 0,
    return_report: bool = False,
):
    """Detect collective events in an image stack directly from the continuous signal.

    Arguments:
        stack (np.ndarray): (T, Y, X) raw signal.
        epsilon (float): number of false events one is willing to expect per movie. Default 1.
        tile (int | str): tile size in pixels, or "auto": the distance beyond which frame-to-frame changes stop
            being correlated; 1.5x and 2x that size are tried as well and the size with the most events is kept
            (epsilon split over the sizes tried).
        mask (np.ndarray | str | None): bool (Y, X) foreground, "auto" (Otsu if the object clearly stands out),
            or None (everything).
        remove_population_trend, baseline_window, allow_merges, refine_start, prune_early, calibration, n_null,
            seed: as in detect_coactivation_events.
        return_report (bool): also return a CoactivationReport (with the tile size used).

    Returns:
        np.ndarray: (T, Y, X) int32 event ids, 0 = background (and the report if return_report).
    """
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError("stack must have shape (T, Y, X)")
    if isinstance(mask, str) and mask == "auto":
        mask = auto_mask(stack)
    elif mask is None:
        mask = np.ones(stack.shape[1:], bool)
    t0 = max(2, noise_scale(stack) if tile == "auto" else int(tile))
    sizes = sorted({t0, int(round(1.5 * t0)), 2 * t0}) if tile == "auto" else [t0]
    kw = dict(
        n_null=n_null,
        seed=seed,
        remove_population_trend=remove_population_trend,
        baseline_window=baseline_window,
        allow_merges=allow_merges,
        refine_start=refine_start,
        prune_early=prune_early,
        calibration=calibration,
    )
    best = None
    for tsz in sizes:
        lab, rep = _tiles_once(stack.astype(np.float32), tsz, mask, epsilon=epsilon / len(sizes), **kw)
        if best is None or rep.n_events > best[1].n_events:
            best = (lab, rep)
    best[1].tiles_tried = sizes
    return best if return_report else best[0]
