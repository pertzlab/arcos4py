from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import ot  # POT: Python Optimal Transport

# Optional: if you already use numba/POT, keep these; otherwise remove the decorations/imports.
from numba import njit, prange
from sklearn.neighbors import KDTree


def global_linker(fn):
    """Mark a callable as a global (per-frame) linker.

    This decorator adds a '__linking_scope__' attribute with value 'global' to the
    decorated function, indicating that it operates on entire frames at once rather
    than on individual clusters.

    Arguments:
        fn (callable): The linking function to be marked as global scope.

    Returns:
        callable: The decorated function with '__linking_scope__' attribute set to 'global'.
    """
    setattr(fn, "__linking_scope__", "global")
    return fn


def per_cluster_linker(fn):
    """Mark a callable as a per-cluster linker.

    This decorator adds a '__linking_scope__' attribute with value 'per_cluster' to the
    decorated function, indicating that it operates on individual clusters separately
    rather than processing entire frames at once.

    Arguments:
        fn (callable): The linking function to be marked as per-cluster scope.

    Returns:
        callable: The decorated function with '__linking_scope__' attribute set to 'per_cluster'.
    """
    setattr(fn, "__linking_scope__", "per_cluster")
    return fn


@per_cluster_linker
def brute_force_linking(
    *args,
    cluster_labels: np.ndarray,
    cluster_coordinates: np.ndarray,
    memory_cluster_labels: np.ndarray,
    memory_kdtree: KDTree,
    epsPrev: float,
    max_cluster_label: int,
    **kwargs: Dict[str, Any],
) -> Tuple[np.ndarray, int]:
    """Link clusters using brute force nearest neighbor approach.

    This per-cluster linking function finds the nearest neighbor in the memory
    frame for each point in the current cluster and assigns cluster labels based
    on proximity within a specified epsilon threshold.

    Arguments:
        cluster_labels (np.ndarray): Labels of points in the current cluster.
        cluster_coordinates (np.ndarray): Coordinates of points in the current cluster, shape (n_points, n_dims).
        memory_cluster_labels (np.ndarray): Labels of points from the previous/memory frame.
        memory_kdtree (KDTree): KDTree built from memory frame coordinates for efficient nearest neighbor queries.
        epsPrev (float): Maximum distance threshold for linking clusters.
        max_cluster_label (int): Current maximum cluster label, used for generating new labels.
        *args: Additional positional arguments (unused).
        **kwargs: Additional keyword arguments (unused).

    Returns:
        Tuple[np.ndarray, int]: Array of new cluster labels for the input cluster and updated maximum cluster label.
    """
    nn_dist, nn_indices = memory_kdtree.query(cluster_coordinates, k=1)
    nn_dist = nn_dist.flatten()
    nn_indices = nn_indices.flatten()

    prev_cluster_labels = memory_cluster_labels[nn_indices]
    prev_cluster_labels_eps = prev_cluster_labels[(nn_dist <= epsPrev)]
    if prev_cluster_labels_eps.size < 1:
        max_cluster_label += 1
        return np.repeat(max_cluster_label, cluster_labels.size), max_cluster_label

    prev_clusternbr_eps_unique = np.unique(prev_cluster_labels_eps, return_index=False)
    if prev_clusternbr_eps_unique.size == 0:
        max_cluster_label += 1
        return np.repeat(max_cluster_label, cluster_labels.size), max_cluster_label

    # Note: per original behavior, propagate prev labels to all points in this cluster
    # once at least one neighbour within eps exists.
    return prev_cluster_labels, max_cluster_label


@njit(parallel=True)
def _compute_filtered_distances(current_coords, memory_coords):
    n, m = len(current_coords), len(memory_coords)
    distances = np.empty((n, m))
    for i in prange(n):
        for j in range(m):
            diff = current_coords[i] - memory_coords[j]
            s = 0.0
            for k in range(diff.shape[0]):
                s += diff[k] * diff[k]
            distances[i, j] = s
    return np.sqrt(distances)


@per_cluster_linker
def transportation_linking(
    *args,
    cluster_labels: np.ndarray,
    cluster_coordinates: np.ndarray,
    memory_cluster_labels: np.ndarray,
    memory_coordinates: np.ndarray,
    memory_kdtree: KDTree,
    epsPrev: float,
    max_cluster_label: int,
    reg: float = 1,
    reg_m: float = 10,
    cost_threshold: float = 0,
    **kwargs: Dict[str, Any],
) -> Tuple[np.ndarray, int]:
    """Link clusters using optimal transport.

    This per-cluster linking function uses unbalanced optimal transport to find
    optimal assignments between points in the current cluster and points from
    the memory frame within a specified distance threshold.

    Arguments:
        cluster_labels (np.ndarray): Labels of points in the current cluster.
        cluster_coordinates (np.ndarray): Coordinates of points in the current cluster, shape (n_points, n_dims).
        memory_cluster_labels (np.ndarray): Labels of points from the previous/memory frame.
        memory_coordinates (np.ndarray): Coordinates of points from the previous/memory frame, shape (n_memory, n_dims).
        memory_kdtree (KDTree): KDTree built from memory frame coordinates for efficient neighbor queries.
        epsPrev (float): Maximum distance threshold for considering linking candidates.
        max_cluster_label (int): Current maximum cluster label, used for generating new labels.
        reg (float): Entropy regularization parameter for optimal transport. Defaults to 1.
        reg_m (float): Mass regularization parameter for unbalanced optimal transport. Defaults to 10.
        cost_threshold (float): Minimum transport probability threshold for accepting assignments. Defaults to 0.
        *args: Additional positional arguments (unused).
        **kwargs: Additional keyword arguments (unused).

    Returns:
        Tuple[np.ndarray, int]: Array of new cluster labels for the input cluster and updated maximum cluster label.
    """
    neighbors = memory_kdtree.query_radius(cluster_coordinates, r=epsPrev)
    if all(len(ind) == 0 for ind in neighbors):
        max_cluster_label += 1
        return np.full(cluster_labels.shape, max_cluster_label, dtype=int), max_cluster_label

    valid_mem_idx = np.unique(np.concatenate([ind for ind in neighbors if len(ind) > 0]))
    if valid_mem_idx.size == 0:
        max_cluster_label += 1
        return np.full(cluster_labels.shape, max_cluster_label, dtype=int), max_cluster_label

    curr_coords = cluster_coordinates
    mem_coords = memory_coordinates[valid_mem_idx]
    cost_matrix = _compute_filtered_distances(curr_coords, mem_coords)

    n_curr, n_mem = curr_coords.shape[0], mem_coords.shape[0]
    a = np.ones(n_curr) / n_curr
    b = np.ones(n_mem) / n_mem

    # Unbalanced OT
    ot_plan = ot.unbalanced.sinkhorn_unbalanced(a, b, cost_matrix, reg, reg_m)

    best_mem = np.argmax(ot_plan, axis=1)
    probs = ot_plan[np.arange(n_curr), best_mem]
    best_mem[probs < cost_threshold] = -1

    new_cluster_labels = np.full(n_curr, -1, dtype=int)
    for i, m in enumerate(best_mem):
        if m != -1:
            new_cluster_labels[i] = int(memory_cluster_labels[valid_mem_idx[m]])

    if np.any(new_cluster_labels == -1):
        max_cluster_label += 1
        new_cluster_labels[new_cluster_labels == -1] = max_cluster_label

    return new_cluster_labels, max_cluster_label


@global_linker
def global_transportation_linking(
    *,
    frame_cluster_labels: np.ndarray,
    frame_coordinates: np.ndarray,
    memory_cluster_labels: np.ndarray,
    memory_coordinates: np.ndarray,
    memory_kdtree: KDTree,
    epsPrev: float,
    max_cluster_label: int,
    reg: float = 1,
    reg_m: float = 10,
    cost_threshold: float = 0,
    **kwargs: Any,
) -> Tuple[np.ndarray, int]:
    """Perform frame-wise optimal transport linking for all clusters simultaneously.

    This global linking function uses unbalanced optimal transport to find optimal
    assignments between all points in the current frame and candidate points from
    the memory frame. It processes all clusters in a frame at once rather than
    individually.

    Arguments:
        frame_cluster_labels (np.ndarray): Labels of all points in the current frame.
        frame_coordinates (np.ndarray): Coordinates of all points in the current frame, shape (n_points, n_dims).
        memory_cluster_labels (np.ndarray): Labels of points from the previous/memory frame.
        memory_coordinates (np.ndarray): Coordinates of points from the previous/memory frame, shape (n_memory, n_dims).
        memory_kdtree (KDTree): KDTree built from memory frame coordinates for efficient neighbor queries.
        epsPrev (float): Maximum distance threshold for considering linking candidates.
        max_cluster_label (int): Current maximum cluster label, used for generating new labels.
        reg (float): Entropy regularization parameter for optimal transport. Defaults to 1.
        reg_m (float): Mass regularization parameter for unbalanced optimal transport. Defaults to 10.
        cost_threshold (float): Minimum transport probability threshold for accepting assignments. Defaults to 0.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        Tuple[np.ndarray, int]: Array of new cluster labels for all points in the frame and updated maximum cluster label.
    """
    neighbors = memory_kdtree.query_radius(frame_coordinates, r=epsPrev)
    if all(len(ind) == 0 for ind in neighbors):
        max_cluster_label += 1
        return np.full_like(frame_cluster_labels, max_cluster_label), max_cluster_label

    valid_mem_idx = np.unique(np.concatenate([ind for ind in neighbors if len(ind) > 0]))
    if valid_mem_idx.size == 0:
        max_cluster_label += 1
        return np.full_like(frame_cluster_labels, max_cluster_label), max_cluster_label

    curr_coords = frame_coordinates
    mem_coords = memory_coordinates[valid_mem_idx]
    cost_matrix = _compute_filtered_distances(curr_coords, mem_coords)

    n_curr, n_mem = curr_coords.shape[0], mem_coords.shape[0]
    a = np.ones(n_curr) / n_curr
    b = np.ones(n_mem) / n_mem

    ot_plan = ot.unbalanced.sinkhorn_unbalanced(a, b, cost_matrix, reg, reg_m)

    best_mem = np.argmax(ot_plan, axis=1)
    scores = ot_plan[np.arange(n_curr), best_mem]
    best_mem[scores < cost_threshold] = -1

    new_labels = np.full(n_curr, -1, dtype=int)
    for i, m in enumerate(best_mem):
        if m != -1:
            new_labels[i] = int(memory_cluster_labels[valid_mem_idx[m]])

    # Cluster-wise post-processing to mirror per-cluster behavior
    next_max = max_cluster_label
    for cl in np.unique(frame_cluster_labels):
        mask = frame_cluster_labels == cl
        if np.all(new_labels[mask] == -1):
            next_max += 1
            new_labels[mask] = next_max
        elif np.any(new_labels[mask] == -1):
            next_max += 1
            new_labels[np.logical_and(mask, new_labels == -1)] = next_max

    return new_labels, next_max
