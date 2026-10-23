"""Neighbour graphs for co-activation event detection.

CellGraph: tracked cells, one pruned Delaunay graph per frame over the cells present in that frame.
GridGraph: fixed 8-neighbour grid (tiles of an image), the same graph in every frame.

Both expose the per-frame adjacency as scipy CSR matrices (A_t) and as flat arrays (indptr, indices) for
the numba kernels.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay, QhullError


class _Graph:
    T: int
    n: int
    observed: np.ndarray
    A_t: list
    static_A: sparse.csr_matrix | None = None

    def _build_arrays(self):
        """Per-frame CSR as arrays: indptr (F, n + 1) global offsets into indices; F = 1 for a static graph."""
        mats = [self.static_A] if self.static_A is not None else self.A_t
        indptr = np.zeros((len(mats), self.n + 1), np.int64)
        chunks, off = [], 0
        for f, A in enumerate(mats):
            A = A.tocsr()
            A.sort_indices()
            indptr[f] = A.indptr.astype(np.int64) + off
            chunks.append(A.indices.astype(np.int64))
            off += A.indices.size
        self.indptr = indptr
        self.indices = np.concatenate(chunks) if chunks else np.zeros(0, np.int64)


def _simplices(pts):
    """Delaunay simplices; degenerate frames (collinear, coplanar) are joggled, None if that fails too."""
    try:
        return Delaunay(pts).simplices
    except QhullError:
        try:
            return Delaunay(pts, qhull_options="QJ").simplices
        except QhullError:
            return None


class CellGraph(_Graph):
    """Tracked cells: one Delaunay graph per frame over the cells present.

    Long edges are pruned with one Tukey fence (Q3 + 1.5 IQR of all edge lengths in the movie).
    """

    def __init__(self, pos_t, observed):
        """Build the graphs from positions (T, n, d) and presence (T, n); d = 2 or 3."""
        dim = pos_t.shape[-1]
        if dim < 2:
            raise ValueError("Co-activation detection needs at least 2 position columns")
        self.T, self.n = observed.shape
        self.observed = observed
        pairs = list(combinations(range(dim + 1), 2))  # every edge of a simplex (triangle, tetrahedron)
        lengths, edges_t = [], []
        for t in range(self.T):
            idx = np.flatnonzero(observed[t])
            if idx.size < dim + 2:
                edges_t.append(np.zeros((0, 3)))
                continue
            sm = _simplices(pos_t[t, idx])
            if sm is None:
                edges_t.append(np.zeros((0, 3)))
                continue
            e = np.sort(np.vstack([sm[:, list(p)] for p in pairs]), axis=1).astype(np.int64)
            key = np.unique(e[:, 0] * idx.size + e[:, 1])
            e = np.c_[key // idx.size, key % idx.size]
            L = np.linalg.norm(pos_t[t, idx[e[:, 0]]] - pos_t[t, idx[e[:, 1]]], axis=1)
            edges_t.append(np.c_[idx[e], L])
            lengths.append(L)
        if lengths:
            q1, q3 = np.quantile(np.concatenate(lengths), [0.25, 0.75])
            fence = q3 + 1.5 * (q3 - q1)
        else:
            fence = np.inf
        self.A_t = []
        for e in edges_t:
            if e.shape[0] == 0:
                self.A_t.append(sparse.csr_matrix((self.n, self.n), dtype=bool))
                continue
            e = e[e[:, 2] <= fence][:, :2].astype(int)
            A = sparse.coo_matrix((np.ones(len(e), bool), (e[:, 0], e[:, 1])), shape=(self.n, self.n))
            self.A_t.append((A + A.T).tocsr())
        self._build_arrays()


class GridGraph(_Graph):
    """Fixed 8-neighbour grid of tiles (the same graph in every frame)."""

    def __init__(self, keep_yx, T):
        """Build the grid from the (y, x) tile indices kept and the number of frames."""
        self.T, self.n = T, len(keep_yx)
        self.observed = np.ones((T, self.n), bool)
        index = {tuple(p): k for k, p in enumerate(map(tuple, keep_yx))}
        r, c = [], []
        for k, (y, x) in enumerate(keep_yx):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    j = index.get((y + dy, x + dx))
                    if j is not None and j != k:
                        r.append(k)
                        c.append(j)
        A = sparse.coo_matrix((np.ones(len(r), bool), (r, c)), shape=(self.n, self.n)).tocsr()
        self.static_A = A
        self.A_t = [A] * T
        self._build_arrays()
