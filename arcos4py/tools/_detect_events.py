"""Module to track and detect collective events.

Example:
    >>> from arcos4py.tools import track_events_image
    >>> ts = track_events_image(data)

    >>> from arcos4py.tools import track_events_dataframe
    >>> ts = track_events_dataframe(data)
"""

from __future__ import annotations

import functools
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
from kneed import KneeLocator
from numba import njit, prange
from skimage.transform import rescale
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neighbors import KDTree
from tqdm import tqdm

from ..plotting._plotting import LineagePlot
from ..tools._arcos4py_deprecation import handle_deprecated_params

AVAILABLE_CLUSTERING_METHODS = ['dbscan', 'hdbscan']
AVAILABLE_LINKING_METHODS = ['nearest', 'transportation']


def downscale_image(image, scale_factor):
    """Downscale a binary image by a given scale factor.

    Parameters:
    image: np.array
        Input binary image
    scale_factor: float
        The scale factor for downscaling the image

    Returns:
    downscaled_image: np.array
        The downscaled binary image
    """
    # Since the input is binary, we want to use the mode 'reflect' to keep the binary values
    # Order 0 is Nearest-neighbor sampling, suitable for binary images.
    if scale_factor == 1:
        return image
    scale_factor = 1 / scale_factor
    downscaled_image = rescale(image, scale_factor, mode='reflect', order=0, anti_aliasing=False)

    # Threshold to convert back to binary
    downscaled_image = (downscaled_image > 0.5).astype(np.uint8)

    return downscaled_image


def upscale_image(image, scale_factor):
    """Upscale a label image by a given scale factor.

    Parameters:
    image: np.array
        Input label image
    scale_factor: float
        The scale factor for upscaling the image

    Returns:
    upscaled_image: np.array
        The upscaled label image
    """
    # Since the input is a label image, we want to use the mode 'reflect'
    # Order 0 is Nearest-neighbor sampling, suitable for label images.
    upscaled_image = rescale(image, scale_factor, mode='reflect', order=0, anti_aliasing=False)

    # Round and cast to int to keep labels intact
    upscaled_image = np.round(upscaled_image).astype(int)

    return upscaled_image


def _group_data(frame_data):
    unique_frame_vals, unique_frame_indices = np.unique(frame_data, axis=0, return_index=True)
    return unique_frame_vals.astype(np.int32), unique_frame_indices[1:]


def _group_array(group_by, *args, return_group_by=True):
    group_by_sort_key = np.argsort(group_by)
    group_by_sorted = group_by[group_by_sort_key]
    _, group_by_cluster_id = _group_data(group_by_sorted)

    result = [group_by_sort_key]

    if return_group_by:
        result.append(np.split(group_by_sorted, group_by_cluster_id))

    for arg in args:
        assert len(arg) == len(group_by), "All arguments must have the same length as group_by."
        arg_sorted = arg[group_by_sort_key]
        result.append(np.split(arg_sorted, group_by_cluster_id))

    return tuple(result)


def _dbscan(x: np.ndarray, eps: float, minClSz: int, n_jobs: int = 1) -> np.ndarray:
    """Dbscan method to run and merge the cluster id labels to the original dataframe.

    Arguments:
        x (np.ndarray): With unique frame and position columns.

    Returns:
        list[np.ndarray]: list with added collective id column detected by DBSCAN.
    """
    if x.size:
        db_array = DBSCAN(eps=eps, min_samples=minClSz, algorithm="kd_tree", n_jobs=n_jobs).fit(x)
        cluster_labels = db_array.labels_
        cluster_list = np.where(cluster_labels > -1, cluster_labels + 1, np.nan)
        return cluster_list

    return np.empty((0, 0))


def _hdbscan(
    x: np.ndarray, eps: float, minClSz: int, min_samples: int | None, cluster_selection_method: str, n_jobs: int = 1
) -> np.ndarray:
    """Hdbscan method to run and merge the cluster id labels to the original dataframe.

    Arguments:
        x (np.ndarray): With unique frame and position columns.

    Returns:
        list[np.ndarray]: list with added collective id column detected by HDBSCAN.
    """
    if x.size:
        db_array = HDBSCAN(
            min_cluster_size=minClSz,
            min_samples=min_samples,
            cluster_selection_epsilon=eps,
            cluster_selection_method=cluster_selection_method,
            n_jobs=n_jobs,
        ).fit(x)
        cluster_labels = db_array.labels_
        cluster_list = np.where(cluster_labels > -1, cluster_labels + 1, np.nan)
        return cluster_list

    return np.empty((0, 0))


def brute_force_linking(
    cluster_labels: np.ndarray,
    cluster_coordinates: np.ndarray,
    memory_cluster_labels: np.ndarray,
    memory_kdtree: KDTree,
    epsPrev: float,
    max_cluster_label: int,
) -> Tuple[np.ndarray, int]:
    """Brute force linking of clusters across frames.

    Arguments:
        cluster_labels (np.ndarray): The cluster labels for the current frame.
        cluster_coordinates (np.ndarray): The cluster coordinates for the current frame.
        memory_cluster_labels (np.ndarray): The cluster labels for previous frames.
        memory_kdtree (KDTree): KDTree for the previous frame's clusters.
        epsPrev (float): Frame-to-frame distance, used to connect clusters across frames.
        max_cluster_label (int): The maximum label for clusters.
        n_jobs (int): The number of parallel jobs.

    Returns:
        Tuple containing the updated cluster labels and the maximum cluster label.
    """
    # calculate nearest neighbour between previoius and current frame
    nn_dist, nn_indices = memory_kdtree.query(cluster_coordinates, k=1)
    nn_dist = nn_dist.flatten()
    nn_indices = nn_indices.flatten()

    prev_cluster_labels = memory_cluster_labels[nn_indices]
    prev_cluster_labels_eps = prev_cluster_labels[(nn_dist <= epsPrev)]
    # only continue if neighbours
    # were detected within eps distance
    if prev_cluster_labels_eps.size < 1:
        max_cluster_label += 1
        return np.repeat(max_cluster_label, cluster_labels.size), max_cluster_label

    prev_clusternbr_eps_unique = np.unique(prev_cluster_labels_eps, return_index=False)

    if prev_clusternbr_eps_unique.size == 0:
        max_cluster_label += 1
        return np.repeat(max_cluster_label, cluster_labels.size), max_cluster_label

    # propagate cluster id from previous frame
    cluster_labels = prev_cluster_labels
    return cluster_labels, max_cluster_label


@njit(parallel=True)
def _compute_filtered_distances(current_coords, memory_coords):
    n, m = len(current_coords), len(memory_coords)
    distances = np.empty((n, m))
    for i in prange(n):
        for j in prange(m):
            distances[i, j] = np.sum((current_coords[i] - memory_coords[j]) ** 2)
    return np.sqrt(distances)


@njit
def _assign_labels(matches, current_indices, memory_indices, memory_cluster_labels, cluster_labels_size):
    new_cluster_labels = np.full(cluster_labels_size, -1)
    for i, m in enumerate(matches):
        if m != -1:
            new_cluster_labels[current_indices[i]] = memory_cluster_labels[memory_indices[m]]
    return new_cluster_labels


def transportation_linking(
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
    """Optimized transportation linking of clusters across frames, using a pre-constructed sklearn KDTree.

    Args:
        cluster_labels (np.ndarray): The cluster labels for the current frame.
        cluster_coordinates (np.ndarray): The cluster coordinates for the current frame.
        memory_cluster_labels (np.ndarray): The cluster labels for previous frames.
        memory_coordinates (np.ndarray): The coordinates for previous frames.
        memory_kdtree (KDTree): Pre-constructed sklearn KDTree for memory coordinates.
        epsPrev (float): Frame-to-frame distance, used to connect clusters across frames.
        max_cluster_label (int): The maximum label for clusters.
        reg (float): Entropy regularization parameter for Sinkhorn algorithm.
        reg_m (float): Marginal relaxation parameter for unbalanced OT.
        cost_threshold (float): Threshold for filtering low-probability matches.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple[np.ndarray, int]: Updated cluster labels and the maximum cluster label.
    """
    # Find neighbors within the maximum allowed distance (epsPrev)
    indices = memory_kdtree.query_radius(cluster_coordinates, r=epsPrev)

    if all(len(ind) == 0 for ind in indices):
        max_cluster_label += 1
        return np.full_like(cluster_labels, max_cluster_label), max_cluster_label

    # Prepare indices of valid points
    valid_mask = np.array([len(ind) > 0 for ind in indices])
    current_indices = np.arange(len(indices))[valid_mask]
    memory_indices = np.array([ind[0] for ind in indices if len(ind) > 0])

    if len(current_indices) == 0:
        max_cluster_label += 1
        return np.full_like(cluster_labels, max_cluster_label), max_cluster_label

    # Compute distance matrix for valid pairs
    filtered_distances = _compute_filtered_distances(
        cluster_coordinates[current_indices], memory_coordinates[memory_indices]
    )

    # Uniform distribution on the valid points
    a = np.ones(len(current_indices)) / len(current_indices)
    b = np.ones(len(memory_indices)) / len(memory_indices)

    # Solve the unbalanced OT problem
    ot_plan = ot.unbalanced.sinkhorn_unbalanced(a, b, filtered_distances, reg, reg_m)

    # Propagate cluster id from previous frame
    matches = np.argmax(ot_plan, axis=1)

    # Set matches to -1 if the cost is too high
    matches[ot_plan[np.arange(len(matches)), matches] < cost_threshold] = -1

    new_cluster_labels = _assign_labels(
        matches, current_indices, memory_indices, memory_cluster_labels, cluster_labels.size
    )

    # Assign new labels to unmatched clusters
    if np.any(new_cluster_labels == -1):
        max_cluster_label += 1
        new_cluster_labels[new_cluster_labels == -1] = max_cluster_label

    return new_cluster_labels, max_cluster_label


@dataclass
class Memory:
    """Memory class for retaining coordinates and cluster IDs over a specified number of time points.

    Attributes:
        n_timepoints (int): The number of time points to retain in memory. Defaults to 1.
        coordinates (List[np.ndarray]): A list of NumPy arrays containing coordinates.
        prev_cluster_ids (List[np.ndarray]): A list of NumPy arrays containing previous cluster IDs.
        max_prev_cluster_id (int): The maximum previous cluster ID.

    Methods:
        update(coordinates, cluster_ids): Updates the coordinates and previous cluster IDs.
        add_timepoint(coordinates, cluster_ids): Appends new coordinates and cluster IDs to the memory.
        remove_timepoint(): Removes a time point if the length of coordinates exceeds n_timepoints.
        reset(): Clears the coordinates and previous cluster IDs.
        all_coordinates: Property that concatenates all coordinates in memory.
        all_cluster_ids: Property that concatenates all cluster IDs in memory.
    """

    n_timepoints: int = 1
    coordinates: list[np.ndarray] = field(default_factory=lambda: [], init=False)
    prev_cluster_ids: list[np.ndarray] = field(default_factory=lambda: [], init=False)
    max_prev_cluster_id: int = 0

    def remove_invalid_ids(self, invalid_ids: set[int]):
        """Remove all points with invalid IDs from memory."""
        for i in range(len(self.prev_cluster_ids)):
            valid_mask = ~np.isin(self.prev_cluster_ids[i], list(invalid_ids))
            self.coordinates[i] = self.coordinates[i][valid_mask]
            self.prev_cluster_ids[i] = self.prev_cluster_ids[i][valid_mask]

    def update(self, new_coordinates, new_cluster_ids):
        """Updates the coordinates and previous cluster IDs.

        Arguments:
            new_coordinates (np.ndarray): The new coordinates.
            new_cluster_ids (np.ndarray): The new cluster IDs.
        """
        self.remove_timepoint()
        self.add_frame(new_coordinates, new_cluster_ids)

    def add_frame(self, new_coordinates, new_cluster_ids):
        """Appends new coordinates and cluster IDs to the memory.

        Arguments:
            new_coordinates (np.ndarray): The new coordinates.
            new_cluster_ids (np.ndarray): The new cluster IDs.
        """
        # remove > 0 cluster ids
        new_coordinates = new_coordinates[new_cluster_ids > 0]
        new_cluster_ids = new_cluster_ids[new_cluster_ids > 0]

        self.coordinates.append(new_coordinates)
        self.prev_cluster_ids.append(new_cluster_ids)

    def remove_timepoint(self):
        """Removes a time point if the length of coordinates exceeds n_timepoints."""
        if len(self.coordinates) > self.n_timepoints:
            self.coordinates.pop(0)
            self.prev_cluster_ids.pop(0)

    def reset(self):
        """Resets the memory."""
        self.coordinates = []
        self.prev_cluster_ids = []

    @property
    def all_coordinates(self):
        """Returns all coordinates in memory as one array."""
        if len(self.coordinates) > 1:
            return np.concatenate(self.coordinates)
        return self.coordinates[0]

    @property
    def all_cluster_ids(self):
        """Returns all cluster IDs in memory as one array."""
        if len(self.prev_cluster_ids) > 1:
            return np.concatenate(self.prev_cluster_ids)
        return self.prev_cluster_ids[0]


class Predictor:
    """Predictor class for predicting future coordinates based on given coordinates and cluster IDs.

    Attributes:
        predictor (Callable): A callable object representing the prediction logic,
            by default the default_predictor is used.
            which predicts coordinates based on centroid displacement.
        prediction_map (Dict[int, float]): A dictionary that maps cluster_ids to coordinates,
            representing coordinate predictions.

    Methods:
        with_default_predictor(): Class method that returns an instance of the Predictor class
            with the default predictor.
        default_predictor(coordinates, cluster_ids): Static method that contains the default prediction logic.
            Predicts coordinates based on centroid displacement.
        predict(coordinates, cluster_ids): Predicts the coordinates for given clusters.
            Requires that the predictor has been fitted.
        fit(coordinates, cluster_ids): Fits the predictor using the given coordinates and cluster IDs.
    """

    def __init__(self, predictor: Callable):
        """Initializes the Predictor class. Defaults to the default_predictor.

        Arguments:
            predictor (Callable): A callable object representing the prediction logic,
                by default the default_predictor is used. See default_predictor for more information.
                Predictor function should take a dictionary that maps cluster_ids to coordinates, and
                return a dictionary that maps cluster_ids to coordinates, representing coordinate predictions.
        """
        self.predictor = predictor if predictor is not None else self.default_predictor
        self.prediction_map: Dict[int, float] = defaultdict()
        self._fitted = False

    @classmethod
    def with_default_predictor(cls):
        """Class method that returns an instance of the Predictor class with the default predictor."""
        return cls(cls.default_predictor)

    @staticmethod
    def default_predictor(cluster_map: Dict[float, Dict[int, Tuple[np.ndarray, Tuple[np.ndarray]]]]):
        """Static method that contains the default prediction logic.

        Predicts coordinates based on centroid displacement for each cluster.

        Arguments:
            cluster_map (Dict[float, Dict[int, Tuple[np.ndarray, Tuple[np.ndarray]]]]):
                A dictionary that maps cluster_ids to coordinates.
        """
        prediction_map: Dict[float, np.ndarray] = defaultdict()

        def _get_centroid(coords: np.ndarray) -> np.ndarray:
            if coords.shape[0] < 2:
                return coords
            return np.mean(coords, keepdims=True, axis=0)

        def _get_velocity(centroids: List[np.ndarray]) -> np.ndarray:
            if len(centroids) < 2:
                return np.zeros_like(centroids)
            return np.mean(np.diff(centroids, axis=0), axis=0)

        for cluster in cluster_map:
            centroids = [_get_centroid(coords) for coords, _ in cluster_map[cluster].values()]
            velocity = _get_velocity(centroids)
            prediction_map[cluster] = velocity

        return prediction_map

    def _create_cluster_map(
        self, coordinates: List[np.ndarray], cluster_ids: List[np.ndarray]
    ) -> Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]]:

        result: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)

        for timepoint, (coords, ids) in enumerate(zip(coordinates, cluster_ids), start=-len(coordinates) + 1):
            unique_ids = np.unique(ids)
            if unique_ids.size == 0:
                result[-1][timepoint] = (np.empty((0, 2)), np.empty((0,)))
            for unique_id in unique_ids:
                id_indices = np.where(ids == unique_id)[0]
                result[unique_id][timepoint] = (coords[id_indices], id_indices)
        return result

    def predict(self, coordinates: List[np.ndarray], cluster_ids: List[np.ndarray], copy=True):
        """Predicts the coordinates for given clusters. Requires that the predictor has been fitted.

        Arguments:
            coordinates (List[np.ndarray]): A list of coordinates for each time point to predict.
            cluster_ids (List[np.ndarray]): A list of cluster IDs for each time point to predict.
            copy (bool): Whether to copy the coordinates before modifying them in place.

        Returns:
            List[np.ndarray]: A list of predicted coordinates for each time point.
        """
        assert len(coordinates) == len(cluster_ids), "The number of coordinates and cluster IDs must be the same"

        if not self._fitted:
            warnings.warn("Predictor has not been fitted yet")
            return coordinates

        if copy:
            coordinates = [coords.copy() for coords in coordinates]

        for coords, ids in zip(coordinates, cluster_ids):
            self._predict_frame(coords, ids)

        return coordinates

    def _predict_frame(self, coordinates: np.ndarray, cluster_ids: np.ndarray):
        # modify coordinates in place
        unique_ids = np.unique(cluster_ids)
        for unique_id in unique_ids:
            if unique_id not in self.prediction_map:
                continue
            id_indices = np.where(cluster_ids == unique_id)
            coordinates[id_indices] = np.add(coordinates[id_indices], self.prediction_map[unique_id])

    def fit(self, coordinates: List[np.ndarray], cluster_ids: List[np.ndarray]):
        """Fit the predictor to the given coordinates and cluster ID pairs.

        Has to be called before predict can be called.

        Arguments:
            coordinates (List[np.ndarray]): List of coordinates for each timepoint.
            cluster_ids (List[np.ndarray]): List of cluster IDs for each timepoint.
        """
        assert len(coordinates) == len(cluster_ids), "The number of coordinates and cluster IDs must be the same"

        if len(coordinates) < 2:
            raise ValueError("There must be at least 2 timepoints to fit the predictor")

        cluster_map = self._create_cluster_map(coordinates, cluster_ids)

        if self.predictor is not None:
            self.prediction_map = self.predictor(cluster_map)
            self._fitted = True


@dataclass
class ClusterNode:
    """Data class representing a node in the lineage tree.

    Attributes:
        cluster_id (int): The cluster ID.
        minframe (int): The minimum frame number.
        maxframe (int): The maximum frame number.
        parents (List[ClusterNode]): The parent nodes.
        children (List[ClusterNode]): The child nodes.
        lineage_id (int): The lineage ID.

    """

    cluster_id: int
    minframe: int
    maxframe: int
    parents: List['ClusterNode']
    children: List['ClusterNode']
    lineage_id: int  # New attribute to track lineage

    def __init__(self, cluster_id, frame, lineage_id=None):
        """Initializes a ClusterNode object.

        Arguments:
            cluster_id (int): The cluster ID.
            frame (int): The frame number.
            lineage_id (int): The lineage ID.
        """
        self.cluster_id = cluster_id
        self.minframe = frame
        self.maxframe = frame
        self.parents = []
        self.children = []
        self.lineage_id = lineage_id if lineage_id is not None else cluster_id

    def __repr__(self):
        """Returns a string representation of the ClusterNode object."""
        return f"ClusterNode(id={self.cluster_id}, frames={self.minframe}-{self.maxframe}, lineage={self.lineage_id})"

    def __hash__(self):
        """Returns a hash of the ClusterNode object."""
        return hash((self.cluster_id, self.minframe, self.maxframe))

    def __eq__(self, other):
        """Checks if two ClusterNode objects are equal."""
        if not isinstance(other, ClusterNode):
            return False
        return (self.cluster_id, self.minframe, self.maxframe) == (other.cluster_id, other.minframe, other.maxframe)


class LineageTracker:
    """Class to track the lineage of clusters over time.

    Attributes:
        nodes (Dict[int, ClusterNode]): Dictionary of cluster IDs to ClusterNode objects.
        max_parents_count (int): The maximum number of parents for a cluster.

    Methods:
        get_cluster_history(cluster_id): Returns the history of a cluster as a list of paths.
        get_lineage_tree(): Returns the lineage tree as a list of nodes and edges.
        plot(): Plots the lineage tree.
    """

    def __init__(self):
        """Initializes a LineageTracker object."""
        self.nodes: Dict[int, ClusterNode] = {}
        self.max_parents_count = 0

    def _add_frame(self, linked_ids, original_ids, frame):
        for curr_id in set(original_ids):
            if curr_id == -1:  # Skip noise
                continue
            curr_id = int(curr_id)
            if curr_id in self.nodes:
                curr_node = self.nodes[curr_id]
                curr_node.maxframe = frame
            else:
                # For new nodes, determine the lineage during creation
                curr_node = ClusterNode(curr_id, frame)
                self.nodes[curr_id] = curr_node

            # Determine parent nodes
            parent_ids = set(linked_ids[original_ids == curr_id])
            if parent_ids:
                # Handle splits or merges here
                # First, filter out invalid or self-referential IDs
                valid_parents = [int(pid) for pid in parent_ids if pid != -1 and pid != curr_id]

                # Assign lineage during merges
                if len(valid_parents) > 0:
                    # Get the lineage from the parent with the larger number of objects
                    parent_count = {pid: (linked_ids == pid).sum() for pid in valid_parents}
                    parent_with_most_objects = max(parent_count, key=parent_count.get)
                    parent_node = self.nodes[parent_with_most_objects]

                    # Assign lineage from the parent with the most objects
                    curr_node.lineage_id = parent_node.lineage_id

                # Handle parent-child relationships
                for parent_id in valid_parents:
                    if parent_id in self.nodes:
                        parent_node = self.nodes[parent_id]
                    else:
                        parent_node = ClusterNode(parent_id, frame - 1)
                        self.nodes[parent_id] = parent_node

                    if curr_node not in parent_node.children:
                        parent_node.children.append(curr_node)
                    if parent_node not in curr_node.parents:
                        curr_node.parents.append(parent_node)

            # Update max_parents_count if necessary
            self.max_parents_count = max(self.max_parents_count, len(curr_node.parents))

    def get_cluster_history(self, cluster_id: int) -> List[List[Tuple[int, int]]]:
        """Returns the history of a cluster as a list of paths, where each path is a list of (frame, cluster_id) tuples.

        Arguments:
            cluster_id (int): The cluster ID to get the history for.

        Returns:
            List[List[Tuple[int, int]]]: The history of the cluster as a list of paths.
        """
        if cluster_id not in self.nodes:
            return []

        start_node = self.nodes[cluster_id]
        paths = []
        visited = set()

        def dfs(node, current_path):
            if node in visited:
                return
            visited.add(node)

            current_path.append(node)

            if not node.parents:
                paths.append(list(current_path))
            else:
                for parent in node.parents:
                    dfs(parent, current_path)

            current_path.pop()
            visited.remove(node)

        dfs(start_node, deque())

        # Sort paths by frame number
        for path in paths:
            path.sort(key=lambda x: x.minframe)

        return paths

    def get_lineage_tree(self):
        """Returns the lineage tree as a list of nodes and edges.

        Returns:
            Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]: The lineage tree as a list of nodes and edges.
        """
        nodes = [(node.minframe, node.cluster_id) for node in self.nodes.values()]
        edges = []
        for node in self.nodes.values():
            for child in node.children:
                edges.append((node.cluster_id, child.cluster_id))

        return nodes, edges

    def _get_immediate_parents(self, cluster_ids):
        """Returns the immediate parent IDs of the given cluster IDs.

        Args:
        cluster_ids (array-like): Array of cluster IDs to find parents for.

        Returns:
        list of tuples: Each tuple contains parent IDs for a cluster, padded with None if necessary.
        """
        parent_ids = []
        for cluster_id in cluster_ids:
            node = self.nodes.get(cluster_id)  # Get the ClusterNode associated with the cluster ID
            if node:
                # Collect all the parent IDs for the node
                parents = [parent.cluster_id for parent in node.parents]
                # Pad the list with None values if there are fewer parents than max_parents_count
                parents += [None] * (self.max_parents_count - len(parents))
            else:
                # If the node doesn't exist, create a list of None values (no parents found)
                parents = [None] * self.max_parents_count
            parent_ids.append(tuple(parents))  # Append the result as a tuple

        return parent_ids

    def _add_parents_and_lineage_to_df(self, df, cluster_id_column):
        """Adds new columns 'parent_1', 'parent_2', etc., and a 'lineage' column to the given DataFrame.

        Args:
        df (pd.DataFrame): Input DataFrame.
        cluster_id_column (str): Name of the column containing cluster IDs.

        Returns:
        pd.DataFrame: DataFrame with the new parent columns and a lineage column added.
        """
        # Get the immediate parents for each cluster ID in the DataFrame
        parent_ids = self._get_immediate_parents(df[cluster_id_column])

        # Add parent columns
        for i in range(self.max_parents_count):
            column_name = f'parent_{i+1}'
            df[column_name] = [parents[i] for parents in parent_ids]

        # Add lineage column
        df['lineage'] = df[cluster_id_column].apply(
            lambda cid: self.nodes[cid].lineage_id if cid in self.nodes else None
        )

        return df

    def copy(self):
        """Returns a shallow copy of the LineageTracker."""
        new_tracker = LineageTracker()
        new_tracker.nodes = self.nodes.copy()
        new_tracker.max_parents_count = self.max_parents_count
        return new_tracker

    def filter(self, criteria: str = None, min_value: int = None, max_value: int = None, ids: np.ndarray = None):
        """Returns a new LineageTracker with nodes filtered by the given criteria.

        Args:
            criteria (str, optional): Filter criteria. One of:
                - 'event_size': Number of frames an event spans
                - 'lineage_size': Number of clusters in a lineage
                - 'event_duration': Time duration of an event
                - 'lineage_duration': Time duration of a lineage
            min_value (int, optional): Minimum value for the criteria. If None, no lower bound.
            max_value (int, optional): Maximum value for the criteria. If None, no upper bound.
            ids (np.ndarray, optional): Explicit array of cluster IDs to keep

        Returns:
            LineageTracker: A new LineageTracker instance containing only the filtered nodes
        """
        filtered_ids = None

        if ids is not None:
            filtered_ids = set(ids)
        elif criteria:
            if criteria == 'event_size':
                filtered_ids = self._get_size_filtered_ids(min_value, max_value)
            elif criteria == 'lineage_size':
                filtered_ids = self._get_lineage_size_filtered_ids(min_value, max_value)
            elif criteria == 'event_duration':
                filtered_ids = self._get_duration_filtered_ids(min_value, max_value)
            elif criteria == 'lineage_duration':
                filtered_ids = self._get_lineage_duration_filtered_ids(min_value, max_value)
            else:
                raise ValueError(
                    f"Invalid filter criteria '{criteria}'. Choose from: "
                    f"['event_size', 'lineage_size', 'event_duration', 'lineage_duration']"
                )

        if filtered_ids is None:
            return self.copy()

        # Create new tracker with filtered nodes and clean up relationships
        new_tracker = LineageTracker()
        new_tracker.nodes = {
            node_id: self._create_clean_node_copy(self.nodes[node_id])
            for node_id in filtered_ids
            if node_id in self.nodes
        }

        # Update relationships between remaining nodes
        self._update_node_relationships(new_tracker)

        # Update max_parents_count for the new tracker
        new_tracker.max_parents_count = max((len(node.parents) for node in new_tracker.nodes.values()), default=0)

        return new_tracker

    def _create_clean_node_copy(self, node: ClusterNode) -> ClusterNode:
        """Creates a copy of a node with empty parent/child lists but preserved attributes."""
        new_node = ClusterNode(node.cluster_id, node.minframe)
        new_node.maxframe = node.maxframe
        new_node.lineage_id = node.lineage_id
        # Don't copy parents or children - these will be rebuilt
        return new_node

    def _update_node_relationships(self, new_tracker: 'LineageTracker'):
        """Updates parent-child relationships between nodes in the new tracker.

        Only maintains relationships where both parent and child exist in the filtered set.
        """
        for node_id, node in new_tracker.nodes.items():
            original_node = self.nodes[node_id]

            # Update parent relationships
            for parent in original_node.parents:
                if parent.cluster_id in new_tracker.nodes:
                    new_parent = new_tracker.nodes[parent.cluster_id]
                    if node not in new_parent.children:
                        new_parent.children.append(node)
                    if new_parent not in node.parents:
                        node.parents.append(new_parent)

            # Update child relationships
            for child in original_node.children:
                if child.cluster_id in new_tracker.nodes:
                    new_child = new_tracker.nodes[child.cluster_id]
                    if node not in new_child.parents:
                        new_child.parents.append(node)
                    if new_child not in node.children:
                        node.children.append(new_child)

    def _check_bounds(self, value: int, min_value: int = None, max_value: int = None) -> bool:
        """Helper method to check if a value is within bounds."""
        if min_value is not None and value < min_value:
            return False
        if max_value is not None and value > max_value:
            return False
        return True

    def _get_size_filtered_ids(self, min_size: int = None, max_size: int = None):
        """Returns set of node IDs filtered by event size."""
        return {
            node.cluster_id
            for node in self.nodes.values()
            if self._check_bounds(node.maxframe - node.minframe + 1, min_size, max_size)
        }

    def _get_duration_filtered_ids(self, min_duration: int = None, max_duration: int = None):
        """Returns set of node IDs filtered by event duration."""
        return {
            node.cluster_id
            for node in self.nodes.values()
            if self._check_bounds(node.maxframe - node.minframe + 1, min_duration, max_duration)
        }

    def _get_lineage_metrics(self):
        """Calculate metrics for each lineage.

        Returns:
            Tuple[Dict, Dict]: Maps of lineage_id to duration and size
        """
        lineage_min_frames = defaultdict(lambda: float('inf'))
        lineage_max_frames = defaultdict(lambda: float('-inf'))
        lineage_sizes = defaultdict(int)

        # First pass: collect frame ranges and count events
        for node in self.nodes.values():
            lineage_id = node.lineage_id
            lineage_min_frames[lineage_id] = min(lineage_min_frames[lineage_id], node.minframe)
            lineage_max_frames[lineage_id] = max(lineage_max_frames[lineage_id], node.maxframe)
            lineage_sizes[lineage_id] += 1

        # Calculate durations
        lineage_durations = {
            lineage_id: max_frame - min_frame + 1
            for lineage_id, max_frame in lineage_max_frames.items()
            for min_frame in [lineage_min_frames[lineage_id]]
        }

        return lineage_durations, lineage_sizes

    def _get_merge_split_stats(self):
        """Calculate merge and split statistics for each lineage.

        Returns:
            Tuple[Dict, Dict]: Maps of lineage_id to merge and split counts
        """
        lineage_merges = defaultdict(int)
        lineage_splits = defaultdict(int)

        # Second pass: count merges and splits
        for node in self.nodes.values():
            lineage_id = node.lineage_id
            if len(node.parents) > 1:
                lineage_merges[lineage_id] += 1
            if len(node.children) > 1:
                lineage_splits[lineage_id] += 1

        return lineage_merges, lineage_splits

    def _get_lineage_size_filtered_ids(self, min_size: int = None, max_size: int = None):
        """Returns set of node IDs filtered by lineage size.

        A lineage's size is the total count of events with that lineage ID.
        """
        _, lineage_sizes = self._get_lineage_metrics()

        return {
            node.cluster_id
            for node in self.nodes.values()
            if self._check_bounds(lineage_sizes[node.lineage_id], min_size, max_size)
        }

    def _get_lineage_duration_filtered_ids(self, min_duration: int = None, max_duration: int = None):
        """Returns set of node IDs filtered by lineage duration.

        A lineage's duration is the span from the earliest to latest frame of any event
        in that lineage.
        """
        lineage_durations, _ = self._get_lineage_metrics()

        return {
            node.cluster_id
            for node in self.nodes.values()
            if self._check_bounds(lineage_durations[node.lineage_id], min_duration, max_duration)
        }

    def reflect(self, data: Union[pd.DataFrame, np.ndarray], cluster_id_column: str = 'cluster_id'):
        """Returns filtered version of input data based on nodes in this tracker.

        Args:
            data: Input DataFrame or array to filter
            cluster_id_column: Column name containing cluster IDs (for DataFrames)

        Returns:
            Union[pd.DataFrame, np.ndarray]: Filtered data matching current tracker's nodes

        Examples:
            # Chain filters and reflect
            filtered_df = (tracker
                         .filter('event_size', min_value=5)
                         .filter('lineage_size', max_value=8)
                         .reflect(df))
        """
        valid_ids = set(self.nodes.keys())

        if isinstance(data, pd.DataFrame):
            data = data[data[cluster_id_column].isin(valid_ids)]
            # remove any parents and lineage columns
            data = data.drop(columns=[col for col in data.columns if 'parent' in col or col == 'lineage'])
            # add parents and lineage columns
            data = self._add_parents_and_lineage_to_df(data, cluster_id_column)
            return data
        else:  # numpy array
            return np.where(np.isin(data, list(valid_ids)), data, 0)

    def plot(self, **kwargs):
        """Plots the lineage tree.

        Arguments:
            **kwargs: Arguments passed to LineagePlot constructor.
                    See LineagePlot documentation for details.

        Returns:
            LineagePlot: The plot object for customization
        """
        plotter = LineagePlot(**kwargs)
        plotter.draw_tree(self)
        return plotter


class Linker:
    """Linker class to link clusters across frames and detect collective events.

    Attributes:
        event_ids (np.ndarray): The event IDs.
        frame_counter (int): The current frame counter.
        LineageTracker (LineageTracker): The LineageTracker object.

    Methods:
        link(input_coordinates): Links clusters across frames and detects collective events.
        get_event_ids(): Returns the event IDs.
    """

    def __init__(
        self,
        eps: float = 1,
        eps_prev: float | None = None,
        min_clustersize: int = 1,
        min_samples: int | None = None,
        clustering_method: str | Callable = "dbscan",
        linking_method: str = "nearest",
        predictor: bool | Callable = True,
        n_prev: int = 1,
        cost_threshold: float = 0,
        reg: float = 1,
        reg_m: float = 10,
        n_jobs: int = 1,
        allow_merges: bool = False,
        allow_splits: bool = False,
        stability_threshold: int = 10,
        remove_small_clusters: bool = False,
        min_size_for_split: int = 1,
        **kwargs,
    ):
        """Initializes the Linker object.

        Arguments:
            eps (float): The maximum distance between two samples for one to be considered as in
                the neighbourhood of the other.
            eps_prev (float | None): Frame to frame distance, value is used to connect
                collective events across multiple frames. If "None", same value as eps is used.
            min_clustersize (int): The minimum size for a cluster to be identified as a collective event.
            min_samples (int | None): The number of samples (or total weight) in a neighbourhood for a
                point to be considered as a core point. This includes the point itself.
                Only used if clusteringMethod is 'hdbscan'. If None, minSamples =  minClsz.
            clustering_method (str | Callable): The clustering method to be used. One of ['dbscan', 'hdbscan']
                or a callable that takes a 2d array of coordinates and returns a list of cluster labels.
                Arguments `eps`, `minClSz` and `minSamples` are ignored if a callable is passed.
            linking_method (str): The linking method to be used.
            predictor (bool | Callable): The predictor method to be used.
            n_prev (int): Number of previous frames the tracking
                algorithm looks back to connect collective events.
            n_jobs (int): Number of jobs to run in parallel (only for clustering algorithm).
            cost_threshold (int): Threshold for filtering low-probability matches (only for transportation linking).
            reg (float): Entropy regularization parameter for unbalanced OT algorithm (only for transportation linking).
            reg_m (float): Marginal relaxation parameter for unbalanced OT (only for transportation linking).
            stability_threshold (int): Number of consecutive frames a merge/split must persist to be considered stable.
            allow_merges (bool): Whether to allow merges.
            allow_splits (bool): Whether to allow splits.
            remove_small_clusters (bool): Whether to remove clusters smaller than min_clustersize.
            min_size_for_split (int): The minimum size for a cluster to be considered for splitting. Multiple of min_clustersize.
            kwargs (Any): Additional keyword arguments. Includes deprecated parameters for backwards compatibility.
                - epsPrev: Deprecated parameter for eps_prev. Use eps_prev instead.
                - minClSz: Deprecated parameter for min_clustersize. Use min_clustersize instead.
                - minSamples: Deprecated parameter for min_samples. Use min_samples instead.
                - clusteringMethod: Deprecated parameter for clustering_method. Use clustering_method instead.
                - nPrev: Deprecated parameter for n_prev. Use n_prev instead.
                - nJobs: Deprecated parameter for n_jobs. Use n_jobs instead.
        """
        map_params = {
            'epsPrev': 'eps_prev',
            'minClSz': 'min_clustersize',
            'minSamples': 'min_samples',
            'clusteringMethod': 'clustering_method',
            'linkingMethod': 'linking_method',
            'nPrev': 'n_prev',
            'nJobs': 'n_jobs',
        }

        # check for allowed kwargs
        for key in kwargs:
            if key not in map_params.keys():
                raise ValueError(f'Invalid keyword argument {key}')

        # Handle deprecated parameters
        kwargs = handle_deprecated_params(map_params, **kwargs)

        # Assign parameters
        eps_prev = kwargs.get('eps_prev', eps_prev)
        min_clustersize = kwargs.get('min_clustersize', min_clustersize)
        min_samples = kwargs.get('min_samples', min_samples)
        clustering_method = kwargs.get('clustering_method', clustering_method)
        n_prev = kwargs.get('n_prev', n_prev)
        n_jobs = kwargs.get('n_jobs', n_jobs)

        self._predictor: Predictor | None  # for mypy
        self._memory = Memory(n_timepoints=n_prev)

        if callable(predictor):
            self._predictor = Predictor(predictor)
        elif predictor:
            self._predictor = Predictor.with_default_predictor()
        else:
            self._predictor = None

        self._nn_tree: KDTree | None = None
        if eps_prev is None:
            self._eps_prev = eps
        else:
            self._eps_prev = eps_prev

        self._reg = reg
        self._reg_m = reg_m
        self._cost_threshold = cost_threshold

        self._n_jobs = n_jobs
        self._validate_input(eps, eps_prev, min_clustersize, min_samples, clustering_method, n_prev, n_jobs)

        self.event_ids = np.empty((0, 0), dtype=np.int64)

        if hasattr(clustering_method, '__call__'):  # check if it's callable
            self.clustering_function = clustering_method
        else:
            if clustering_method == "dbscan":
                self.clustering_function = functools.partial(_dbscan, eps=eps, minClSz=min_clustersize)
            elif clustering_method == "hdbscan":
                self.clustering_function = functools.partial(
                    _hdbscan, eps=eps, minClSz=min_clustersize, min_samples=min_samples, cluster_selection_method='eom'
                )
            else:
                raise ValueError(
                    f'Clustering method must be either in {AVAILABLE_CLUSTERING_METHODS} or a callable with data as the only argument an argument'  # noqa E501
                )

        if hasattr(linking_method, '__call__'):  # check if it's callable
            self.linking_function = linking_method
        else:
            if linking_method == "nearest":
                self.linking_function = 'brute_force_linking'
            elif linking_method == "transportation":
                self.linking_function = 'transportation_linking'
            else:
                raise ValueError(
                    f'Linking method must be either in {AVAILABLE_LINKING_METHODS} or a callable'  # noqa E501
                )

        self._stability_threshold = stability_threshold
        self._allow_merges = allow_merges
        self._allow_splits = allow_splits
        self._merge_candidate_history: Dict[int, List[Tuple[List[int], int]]] = {}
        self._split_candidate_history: Dict[int, List[Tuple[int, List[int]]]] = {}
        self.lineage_tracker = LineageTracker()
        self.frame_counter = -1  # Start at -1 to get the first frame to be 0
        self._remove_small_clusters = remove_small_clusters
        self._min_clustersize = min_clustersize
        self._min_size_for_split = min_size_for_split

    def _validate_input(self, eps, eps_prev, min_clustersize, min_samples, clustering_method, n_prev, n_jobs):
        if not isinstance(eps, (int, float, str)):
            raise ValueError(f"eps must be a number or None, got {eps}")
        if not isinstance(eps_prev, (int, float, type(None))):
            raise ValueError(f"{eps_prev} must be a number or None, got {eps_prev}")
        if not isinstance(min_samples, (int, type(None))):
            raise ValueError(f"{min_samples} must be a number or None, got {min_samples}")
        for i in [min_clustersize, n_prev, n_jobs]:
            if not isinstance(i, int):
                raise ValueError(f"{i} must be an int, got {i}")
        if not isinstance(clustering_method, str) and not callable(clustering_method):
            raise ValueError(f"clusteringMethod must be a string or a callable, got {clustering_method}")

    # @profile
    def _clustering(self, x):
        if x.size == 0:
            return np.empty((0,), dtype=np.int64), x, np.empty((0, 1), dtype=bool)
        clusters = self.clustering_function(x)
        nanrows = np.isnan(clusters)
        return clusters[~nanrows], x[~nanrows], nanrows

    # @profile
    def _link_next_cluster(self, cluster: np.ndarray, cluster_coordinates: np.ndarray):
        if self.linking_function == 'brute_force_linking':
            linked_clusters, max_cluster_label = brute_force_linking(
                cluster_labels=cluster,
                cluster_coordinates=cluster_coordinates,
                memory_cluster_labels=self._memory.all_cluster_ids,
                memory_kdtree=self._nn_tree,
                epsPrev=self._eps_prev,
                max_cluster_label=self._memory.max_prev_cluster_id,
            )
        elif self.linking_function == 'transportation_linking':
            linked_clusters, max_cluster_label = transportation_linking(
                cluster_labels=cluster,
                cluster_coordinates=cluster_coordinates,
                memory_cluster_labels=self._memory.all_cluster_ids,
                memory_coordinates=self._memory.all_coordinates,
                memory_kdtree=self._nn_tree,
                epsPrev=self._eps_prev,
                max_cluster_label=self._memory.max_prev_cluster_id,
                reg=self._reg,
                reg_m=self._reg_m,
                cost_threshold=self._cost_threshold,
            )
        else:
            raise ValueError(f'Linking method must be (for now) in {AVAILABLE_LINKING_METHODS}')

        self._memory.max_prev_cluster_id = max_cluster_label

        return linked_clusters

    def _update_tree(self, coords):
        self._nn_tree = KDTree(coords)

    def _get_next_id(self) -> int:
        """Generate a new unique ID."""
        self._memory.max_prev_cluster_id += 1
        if self._memory.max_prev_cluster_id == 92:
            pass
        return self._memory.max_prev_cluster_id

    def _apply_remove_small_clusters(self, linked_cluster_ids, original_cluster_ids):
        # Remove small clusters if necessary
        if self._remove_small_clusters:
            unique_ids, counts = np.unique(linked_cluster_ids, return_counts=True)
            for unique_id, count in zip(unique_ids, counts):
                if count < self._min_clustersize:
                    linked_cluster_ids[linked_cluster_ids == unique_id] = -1  # Mark as noise

        # if subcluster contained in original cluster_ids is too small, mark as value from a directly connected cluster
        for original_id in np.unique(original_cluster_ids):
            mask = original_cluster_ids == original_id
            unique_ids, counts = np.unique(linked_cluster_ids[mask], return_counts=True)
            for unique_id, count in zip(unique_ids, counts):
                if count < self._min_clustersize:
                    linked_cluster_ids[mask & (linked_cluster_ids == unique_id)] = -1  # Mark as noise

        return linked_cluster_ids

    def link(self, input_coordinates: np.ndarray) -> None:
        """Links clusters across frames and detects collective events.

        Arguments:
            input_coordinates (np.ndarray): The input coordinates.
        """
        self.frame_counter += 1
        original_cluster_ids, coordinates, nanrows = self._clustering(input_coordinates)

        if not len(self._memory.prev_cluster_ids):
            linked_cluster_ids = self._update_id_empty(original_cluster_ids)
        elif original_cluster_ids.size == 0 or self._memory.all_cluster_ids.size == 0:
            linked_cluster_ids = self._update_id_empty(original_cluster_ids)
        else:
            linked_cluster_ids = self._update_id(original_cluster_ids, coordinates)

        if self._remove_small_clusters:
            final_cluster_ids = self._apply_remove_small_clusters(linked_cluster_ids, original_cluster_ids)

        # Apply stable merges and splits
        final_cluster_ids = self._apply_stable_merges_splits(linked_cluster_ids, original_cluster_ids)

        # Update lineage graph
        self.lineage_tracker._add_frame(linked_cluster_ids, final_cluster_ids, self.frame_counter)

        # Update memory and fit predictor
        self._memory.add_frame(new_coordinates=coordinates, new_cluster_ids=final_cluster_ids)
        if self._predictor is not None and len(self._memory.coordinates) > 1:
            self._predictor.fit(coordinates=self._memory.coordinates, cluster_ids=self._memory.prev_cluster_ids)
        self._memory.remove_timepoint()

        event_ids = np.full_like(nanrows, -1, dtype=np.int64)
        event_ids[~nanrows] = final_cluster_ids

        self.event_ids = event_ids

    # @profile
    def _update_id(self, cluster_ids, coordinates):
        memory_coordinates = self._memory.coordinates
        memory_cluster_ids = self._memory.prev_cluster_ids

        if self._predictor is not None and self._predictor._fitted:
            memory_coordinates = self._predictor.predict(memory_coordinates, memory_cluster_ids, copy=True)

        if len(memory_coordinates) > 1:
            memory_coordinates = np.concatenate(memory_coordinates)
        elif len(memory_coordinates) == 1:
            memory_coordinates = memory_coordinates[0]
        else:
            raise ValueError("Memory coordinates are empty")

        self._update_tree(memory_coordinates)
        # group by cluster id
        cluster_ids_sort_key, grouped_clusters, grouped_coordinates = _group_array(cluster_ids, coordinates)

        # # do linking
        linked_cluster_ids = [
            self._link_next_cluster(cluster, cluster_coordinates)
            for cluster, cluster_coordinates in zip(grouped_clusters, grouped_coordinates)
        ]
        # restore original data order
        revers_sort_key = np.argsort(cluster_ids_sort_key)
        linked_cluster_ids = np.concatenate(linked_cluster_ids)[revers_sort_key]
        return linked_cluster_ids

    def _update_id_empty(self, cluster_ids):
        linked_cluster_ids = cluster_ids + self._memory.max_prev_cluster_id
        try:
            self._memory.max_prev_cluster_id = np.nanmax(linked_cluster_ids)
        except ValueError:
            pass
        return linked_cluster_ids

    def _identify_potential_merges_splits(self, linked_cluster_ids, original_cluster_ids):
        # filter out noise and 0s
        linked_unique = np.unique(linked_cluster_ids[linked_cluster_ids > 0])
        original_unique = np.unique(original_cluster_ids[original_cluster_ids > 0])

        potential_merges = {}
        potential_splits = {}

        if self._allow_splits:
            for linked_id in np.sort(linked_unique):  # Sort linked_unique
                linked_mask = linked_cluster_ids == linked_id
                original_ids = np.unique(original_cluster_ids[linked_mask])

                if len(original_ids) > 1:
                    potential_splits[linked_id] = sorted(original_ids.tolist())  # Sort original_ids

        if self._allow_merges:
            for original_id in np.sort(original_unique):  # Sort original_unique
                original_mask = original_cluster_ids == original_id
                linked_ids = np.unique(linked_cluster_ids[original_mask])
                # filter out noise and 0s
                linked_ids = linked_ids[linked_ids > 0]

                if len(linked_ids) > 1:
                    potential_merges[original_id] = sorted(linked_ids.tolist())  # Sort linked_ids

        return potential_merges, potential_splits

    def _apply_stable_merges_splits(self, linked_cluster_ids, original_cluster_ids):
        potential_merges, potential_splits = self._identify_potential_merges_splits(
            linked_cluster_ids, original_cluster_ids
        )

        final_cluster_ids = linked_cluster_ids.copy()
        split_merge_events = []
        current_frame = self.frame_counter

        # Process potential splits
        for linked_id, original_ids in sorted(potential_splits.items()):
            split_sizes = [
                np.sum((linked_cluster_ids == linked_id) & (original_cluster_ids == orig_id))
                for orig_id in original_ids
            ]

            if all(size >= self._min_clustersize * self._min_size_for_split for size in split_sizes):
                split_key = linked_id
                if split_key not in self._split_candidate_history:
                    self._split_candidate_history[split_key] = []
                self._split_candidate_history[split_key].append(current_frame)

                # stability condition for splits
                frames = self._split_candidate_history[split_key]
                frames_in_window = [f for f in frames if current_frame - f < self._stability_threshold]
                if len(frames_in_window) >= self._stability_threshold:
                    split_merge_events.append(('split', split_key, original_ids))

        # Process potential merges
        for original_id, linked_ids in sorted(potential_merges.items()):
            merge_key = tuple(sorted(linked_ids))
            if merge_key not in self._merge_candidate_history:
                self._merge_candidate_history[merge_key] = []
            self._merge_candidate_history[merge_key].append(current_frame)

            # stability condition for merges
            frames = self._merge_candidate_history[merge_key]
            frames_in_window = [f for f in frames if current_frame - f < self._stability_threshold]
            if len(frames_in_window) >= self._stability_threshold:
                split_merge_events.append(('merge', merge_key, linked_ids))

        # Track which points have been modified to prevent double-modifications
        modified_points = np.zeros_like(final_cluster_ids, dtype=bool)

        # Track which IDs become invalid due to splits/merges
        invalid_ids = set()

        # Resolve and apply changes
        for event_type, event_key, cluster_ids in sorted(split_merge_events):
            if event_type == 'merge':
                merge_mask = np.zeros_like(final_cluster_ids, dtype=bool)
                for linked_id in cluster_ids:
                    merge_mask |= final_cluster_ids == linked_id
                    invalid_ids.add(linked_id)  # Add all merged IDs to invalid set

                valid_merge_points = merge_mask & ~modified_points
                if np.sum(valid_merge_points) >= self._min_clustersize:
                    merge_id = self._get_next_id()
                    final_cluster_ids[valid_merge_points] = merge_id
                    modified_points[valid_merge_points] = True

            elif event_type == 'split':
                linked_id = event_key
                invalid_ids.add(linked_id)  # Add split ID to invalid set

                for original_id in cluster_ids:
                    split_mask = (linked_cluster_ids == linked_id) & (original_cluster_ids == original_id)
                    valid_split_points = split_mask & ~modified_points
                    if np.sum(valid_split_points) >= self._min_clustersize:
                        split_id = self._get_next_id()
                        final_cluster_ids[valid_split_points] = split_id
                        modified_points[valid_split_points] = True

        # Remove all invalid IDs from memory
        if invalid_ids:
            self._memory.remove_invalid_ids(invalid_ids)

        # Clean up history
        self._merge_candidate_history = {
            k: [f for f in v if current_frame - f < self._stability_threshold]
            for k, v in self._merge_candidate_history.items()
            if v
        }
        self._split_candidate_history = {
            k: [f for f in v if current_frame - f < self._stability_threshold]
            for k, v in self._split_candidate_history.items()
            if v
        }

        return final_cluster_ids


class BaseTracker(ABC):
    """Abstract base class for tracker classes."""

    def __init__(self, linker: Linker):
        """Initializes the BaseTracker object.

        Arguments:
            linker (Linker): The Linker object to use for tracking.
        """
        self.linker = linker

    @abstractmethod
    def track_iteration(self, data):
        """Tracks events in a single frame. Needs to be implemented by subclasses."""
        pass

    @abstractmethod
    def track(self, data: Union[pd.DataFrame, np.ndarray]) -> Generator:
        """Main method for tracking events through the data. Needs to be implemented by subclasses."""
        pass


class DataFrameTracker(BaseTracker):
    """Tracker class for data frames that works in conjunction with the Linker class.

    Methods:
        track_iteration(x: pd.DataFrame):
            Tracks events in a single frame.
        track(x: pd.DataFrame) -> Generator:
            Main method for tracking events through the dataframe. Yields the tracked data frame for each iteration.
    """

    def __init__(
        self,
        linker: Linker,
        position_columns: list[str] = ['x'],
        frame_column: str = 'frame',
        obj_id_column: str | None = None,
        binarized_measurement_column: str | None = None,
        clid_column: str = 'clTrackID',
        **kwargs,
    ):
        """Initializes the DataFrameTracker object.

        Arguments:
            linker (Linker): The Linker object used for linking events.
            position_columns (list[str]): List of strings representing the coordinate columns.
            frame_column (str): String representing the frame/timepoint column in the dataframe.
            obj_id_column (str | None): String representing the ID column, or None if not present. Defaults to None.
            binarized_measurement_column (str | None): String representing the binary measurement column, or None if not present.
                Defaults to None.
            clid_column (str): String representing the collision track ID column. Defaults to 'clTrackID'.
            kwargs (Any): Additional keyword arguments. Includes deprecated parameters for backwards compatibility.
                - coordinates_column: Deprecated parameter for position_columns. Use position_columns instead.
                - collid_column: Deprecated parameter, use clid_column instead.
                - id_column: Deprecated parameter, use obj_id_column instead.
                - bin_meas_column: Deprecated parameter, use binarized_measurement_column instead.
        """
        map_deprecated_params = {
            'coordinates_column': 'position_columns',
            'collid_column': 'clid_column',
            'id_column': 'obj_id_column',
            'bin_meas_column': 'binarized_measurement_column',
        }

        # check for allowed kwargs
        for key in kwargs:
            if key not in map_deprecated_params.keys():
                raise ValueError(f'Invalid keyword argument {key}')

        corrected_kwargs = handle_deprecated_params(map_deprecated_params, **kwargs)

        # Assign parameters
        position_columns = corrected_kwargs.get('position_columns', position_columns)
        obj_id_column = corrected_kwargs.get('obj_id_column', obj_id_column)
        binarized_measurement_column = corrected_kwargs.get(
            'binarized_measurement_column', binarized_measurement_column
        )
        clid_column = corrected_kwargs.get('clid_column', clid_column)

        super().__init__(linker)
        self._coordinates_column = position_columns
        self._frame_column = frame_column
        self._id_column = obj_id_column
        self._binarized_measurement_column = binarized_measurement_column
        self._collid_column = clid_column
        self._validate_input(position_columns, frame_column, obj_id_column, binarized_measurement_column, clid_column)

    def _validate_input(
        self,
        coordinates_column: list[str],
        frame_column: str,
        id_column: str | None,
        bin_meas_column: str | None,
        collid_column: str,
    ):
        necessray_cols: list[Any] = [frame_column, collid_column]
        necessray_cols.extend(coordinates_column)
        optional_cols: list[Any] = [id_column, bin_meas_column]

        for col in necessray_cols:
            if not isinstance(col, str):
                raise TypeError(f'Column names must be of type str, {col} given.')

        for col in optional_cols:
            if not isinstance(col, (str, type(None))):
                raise TypeError(f'Column names must be of type str or None, {col} given.')

    def _select_necessary_columns(
        self,
        data: pd.DataFrame,
        position_columns: list[str],
    ) -> np.ndarray:
        """Select necessary input colums from input data and returns them as numpy arrays.

        Arguments:
            data (DataFrame): Containing necessary columns.
            position_columns (list): string representation of position columns in data.

        Returns:
            np.ndarray, np.ndarray: Filtered columns necessary for calculation.
        """
        pos_columns_np = data[position_columns].to_numpy()
        if pos_columns_np.ndim == 1:
            pos_columns_np = pos_columns_np[:, np.newaxis]
        return pos_columns_np

    def _sort_input(self, x: pd.DataFrame, frame_column: str, object_id_column: str | None) -> pd.DataFrame:
        """Sorts the input dataframe according to the frame column and track id column if available."""
        if object_id_column:
            x = x.sort_values([frame_column, object_id_column]).reset_index(drop=True)
        else:
            x = x.sort_values([frame_column]).reset_index(drop=True)
        return x

    def _filter_active(self, data: pd.DataFrame, binarized_measurement_column: Union[str, None]) -> pd.DataFrame:
        """Selects rows with binary value of greater than 0.

        Arguments:
            data (DataFrame): Dataframe containing necessary columns.
            binarized_measurement_column (str|None): Either name of the binary column or None if no such column exists.

        Returns:
            DataFrame: Filtered pandas DataFrame.
        """
        if binarized_measurement_column is not None:
            data = data[data[binarized_measurement_column] > 0]
        return data

    # @profile
    def track_iteration(self, x: pd.DataFrame) -> pd.DataFrame:
        """Tracks events in a single frame. Returns dataframe with event ids.

        Arguments:
            x (pd.DataFrame): Dataframe to track.

        Returns:
            pd.DataFrame: Dataframe with event ids.
        """
        x_filtered = self._filter_active(x, self._binarized_measurement_column)

        coordinates_data = self._select_necessary_columns(
            x_filtered,
            self._coordinates_column,
        )
        self.linker.link(coordinates_data)

        if self._collid_column in x.columns:
            df_out = x_filtered.drop(columns=[self._collid_column]).copy()
        else:
            df_out = x_filtered.copy()
        event_ids = self.linker.event_ids

        if not event_ids.size:
            df_out[self._collid_column] = 0
            return df_out

        df_out[self._collid_column] = self.linker.event_ids
        if any([self.linker._allow_merges, self.linker._allow_splits]):
            df_out = self.linker.lineage_tracker._add_parents_and_lineage_to_df(
                df_out,
                self._collid_column,
            )
        return df_out

    def track(self, x: pd.DataFrame) -> Generator:
        """Main method for tracking events through the dataframe. Yields the tracked dataframe for each iteration.

        Arguments:
            x (pd.DataFrame): Dataframe to track.

        Yields:
            Generator: Tracked dataframe.
        """
        if x.empty:
            raise ValueError('Input is empty')
        x_sorted = self._sort_input(x, frame_column=self._frame_column, object_id_column=self._id_column)

        for t in range(x_sorted[self._frame_column].min(), x_sorted[self._frame_column].max() + 1):
            x_frame = x_sorted.query(f'{self._frame_column} == {t}')
            x_tracked = self.track_iteration(x_frame)
            yield x_tracked


class ImageTracker(BaseTracker):
    """Tracker class for image data that works in conjunction with the Linker class.

    Methods:
        track_iteration(x: np.ndarray):
            Tracks events in a single frame. Returns the tracked labels.
        track(x: np.ndarray, dims: str = "TXY") -> Generator:
            Main method for tracking events through the image series. Yields the tracked image for each iteration.
    """

    def __init__(self, linker: Linker, downsample: int = 1):
        """Initializes the ImageTracker object.

        Arguments:
            linker (Linker): The Linker object used for linking events.
            downsample (int): Downsampling factor for the images. Defaults to 1, meaning no downsampling.
        """
        super().__init__(linker)
        self._downsample = downsample

    def _image_to_coordinates(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Converts a 2d image series to input that can be accepted by the ARCOS event detection function\
        with columns for x, y, and intensity.

        Arguments:
            image (np.ndarray): Image to convert. Will be coerced to int32.
            dims (str): String of dimensions in order. Default is "TXY". Possible values are "T", "X", "Y", and "Z".
        Returns (tuple[np.ndarray, np.ndarray, np.ndarray]): Tuple of arrays with coordinates, measurements,
            and frame numbers.
        """
        # convert to int16
        image = image.astype(np.uint16)

        coordinates_array = np.moveaxis(np.indices(image.shape), 0, len(image.shape)).reshape((-1, len(image.shape)))
        meas_array = image.flatten()

        return coordinates_array, meas_array

    def _filter_active(self, position_data: np.ndarray, binarized_measurement_column: np.ndarray) -> np.ndarray:
        """Selects rows with binary value of greater than 0.

        Arguments:
            frame_data (np.ndarray): frame column as a numpy array.
            position_data (np.ndarray): positions/coordinate columns as a numpy array.
            binarized_measurement_column (np.ndarray): binary measurement column as a numpy array.

        Returns:
            np.ndarray: Filtered numpy arrays.
        """
        if binarized_measurement_column is not None:
            active = np.argwhere(binarized_measurement_column > 0).flatten()
            position_data = position_data[active]
        return position_data

    def _coordinates_to_image(self, x, position_data, tracked_events):
        # create empty image
        out_img = np.zeros_like(x, dtype=np.uint16)
        if tracked_events.size == 0:
            return out_img
        tracked_events_mask = tracked_events > 0

        position_data = position_data[tracked_events_mask].astype(np.uint16)
        n_dims = position_data.shape[1]

        # Raise an error if dimension is zero
        if n_dims == 0:
            raise ValueError("Dimension of input array not supported.")

        # Create an indexing tuple
        indices = tuple(position_data[:, i] for i in range(n_dims))

        # Index into out_img using the indexing tuple
        out_img[indices] = tracked_events[tracked_events_mask]

        return out_img

    def track_iteration(self, x: np.ndarray) -> np.ndarray:
        """Tracks events in a single frame. Returns the tracked labels.

        Arguments:
            x (np.ndarray): Image to track.

        Returns:
            np.ndarray: Tracked labels.
        """
        x = downscale_image(x, self._downsample)
        coordinates_data, meas_data = self._image_to_coordinates(x)
        coordinates_data_filtered = self._filter_active(coordinates_data, meas_data)

        self.linker.link(coordinates_data_filtered)

        tracked_events = self.linker.event_ids
        out_img = self._coordinates_to_image(x, coordinates_data_filtered, tracked_events)

        if self._downsample > 1:
            out_img = upscale_image(out_img, self._downsample)

        return out_img

    def track(self, x: np.ndarray, dims: str = "TXY") -> Generator:
        """Method for tracking events through the image series. Yields the tracked image for each iteration.

        Arguments:
            x (np.ndarray): Image to track.
            dims (str): String of dimensions in order. Default is "TXY". Possible values are "T", "X", "Y", and "Z".

        Returns:
            Generator: Generator that yields the tracked image for each iteration.
        """
        available_dims = ["T", "X", "Y", "Z"]
        dims_list = list(dims.upper())

        # check input
        for i in dims_list:
            if i not in dims_list:
                raise ValueError(f"Invalid dimension {i}. Must be 'T', 'X', 'Y', or 'Z'.")

        if len(dims_list) > len(set(dims_list)):
            raise ValueError("Duplicate dimensions in dims.")

        if len(dims_list) != x.ndim:
            raise ValueError(
                f"Length of dims must be equal to number of dimensions in image. Image has {x.ndim} dimensions."
            )

        dims_dict = {i: dims_list.index(i) for i in available_dims if i in dims_list}

        # reorder image so T is first dimension
        image_reshaped = np.moveaxis(x, dims_dict["T"], 0)

        for x_frame in image_reshaped:
            x_tracked = self.track_iteration(x_frame)
            yield x_tracked


def track_events_dataframe(
    X: pd.DataFrame,
    position_columns: List[str],
    frame_column: str,
    id_column: str | None = None,
    binarized_measurement_column: str | None = None,
    clid_column: str = "collid",
    eps: float = 1.0,
    eps_prev: float | None = None,
    min_clustersize: int = 3,
    min_samples: int | None = None,
    clustering_method: str = "dbscan",
    linking_method: str = 'nearest',
    allow_merges: bool = False,
    allow_splits: bool = False,
    stability_threshold: int = 10,
    remove_small_clusters: bool = False,
    min_size_for_split: int = 1,
    reg: float = 1,
    reg_m: float = 10,
    cost_threshold: float = 0,
    n_prev: int = 1,
    predictor: bool | Callable = False,
    n_jobs: int = 1,
    show_progress: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Function to track collective events in a dataframe.

    Arguments:
        X (pd.DataFrame): The input dataframe containing the data to track.
        position_columns (List[str]): The names of the columns representing coordinates.
        frame_column (str): The name of the column containing frame ids.
        id_column (str | None): The name of the column representing IDs. None if no such column.
        binarized_measurement_column (str | None): The name of the column representing binarized measurements,
            if None all measurements are used.
        clid_column (str): The name of the output column representing collective events, will be generated.
        eps (float): Maximum distance for clustering, default is 1.
        eps_prev (float | None): Maximum distance for linking previous clusters, if None, eps is used. Default is None.
        min_clustersize (int): Minimum cluster size. Default is 3.
        min_samples (int): The number of samples (or total weight) in a neighbourhood for a
            point to be considered as a core point. This includes the point itself.
            Only used if clusteringMethod is 'hdbscan'. If None, minSamples =  minClsz.
        clustering_method (str): The method used for clustering, one of [dbscan, hdbscan]. Default is "dbscan".
        linking_method (str): The method used for linking, one of ['nearest', 'transportsolver']. Default is 'nearest'.
        allow_merges (bool): Whether or not to allow merges. Default is False.
        allow_splits (bool): Whether or not to allow splits. Default is False.
        stability_threshold (int): Number of frames to consider for stability. Default is 10.
        remove_small_clusters (bool): Whether or not to remove small clusters. Default is False.
        min_size_for_split (int): Minimum size for a split. Default is 1.
        reg (float): Regularization parameter for transportation solver. Default is 1.
        reg_m (float): Regularization parameter for transportation solver. Default is 10.
        cost_threshold (float): Cost threshold for transportation solver. Default is 0.
        n_prev (int): Number of previous frames to consider. Default is 1.
        predictor (bool | Callable): Whether or not to use a predictor. Default is False.
            True uses the default predictor. A callable can be passed to use a custom predictor.
            See default predictor method for details.
        n_jobs (int): Number of jobs to run in parallel. Default is 1.
        show_progress (bool): Whether or not to show progress bar. Default is True.
        **kwargs (Any): Additional keyword arguments. Includes deprecated parameters for backwards compatibility.
            - epsPrev: Deprecated parameter for eps_prev. Use eps_prev instead.
            - minClSz: Deprecated parameter for min_clustersize. Use min_clustersize instead.
            - minSamples: Deprecated parameter for min_samples. Use min_samples instead.
            - clusteringMethod: Deprecated parameter for clustering_method. Use clustering_method instead.
            - linkingMethod: Deprecated parameter for linking_method. Use linking_method instead.
            - nPrev: Deprecated parameter for n_prev. Use n_prev instead.
            - nJobs: Deprecated parameter for n_jobs. Use n_jobs instead.
            - showProgress: Deprecated parameter for show_progress. Use show_progress instead.

    Returns:
        pd.DataFrame: Dataframe with tracked events.
    """
    map_params = {
        "coordinates_column": "position_columns",
        "bin_meas_column": "binarized_measurement_column",
        "collid_column": "clid_column",
        'epsPrev': 'eps_prev',
        'minClSz': 'min_clustersize',
        'minSamples': 'min_samples',
        'clusteringMethod': 'clustering_method',
        'linkingMethod': 'linking_method',
        'nPrev': 'n_prev',
        'nJobs': 'n_jobs',
        'showProgress': 'show_progress',
    }

    # check for allowed kwargs
    for key in kwargs:
        if key not in map_params.keys():
            raise ValueError(f'Invalid keyword argument {key}')

    # Handle deprecated parameters
    kwargs = handle_deprecated_params(map_params, **kwargs)

    # Assign parameters
    eps_prev = kwargs.get('eps_prev', eps_prev)
    min_clustersize = kwargs.get('min_clustersize', min_clustersize)
    min_samples = kwargs.get('min_samples', min_samples)
    clustering_method = kwargs.get('clustering_method', clustering_method)
    linking_method = kwargs.get('linking_method', linking_method)
    n_prev = kwargs.get('n_prev', n_prev)
    n_jobs = kwargs.get('n_jobs', n_jobs)

    linker = Linker(
        eps=eps,
        eps_prev=eps_prev,
        min_clustersize=min_clustersize,
        min_samples=min_samples,
        clustering_method=clustering_method,
        linking_method=linking_method,
        n_prev=n_prev,
        predictor=predictor,
        n_jobs=n_jobs,
        allow_merges=allow_merges,
        allow_splits=allow_splits,
        stability_threshold=stability_threshold,
        remove_small_clusters=remove_small_clusters,
        min_size_for_split=min_size_for_split,
        reg=reg,
        reg_m=reg_m,
        cost_threshold=cost_threshold,
    )

    tracker = DataFrameTracker(
        linker=linker,
        position_columns=position_columns,
        frame_column=frame_column,
        obj_id_column=id_column,
        binarized_measurement_column=binarized_measurement_column,
        clid_column=clid_column,
    )
    df_out = pd.concat(
        [timepoint for timepoint in tqdm(tracker.track(X), total=X[frame_column].nunique(), disable=not show_progress)]
    ).reset_index(drop=True)

    if any([allow_merges, allow_splits]):
        return df_out.query(f"{clid_column} != -1").reset_index(drop=True), linker.lineage_tracker
    return df_out.query(f"{clid_column} != -1").reset_index(drop=True)


def track_events_image(
    X: np.ndarray,
    eps: float = 1,
    eps_prev: float | None = None,
    min_clustersize: int = 1,
    min_samples: int | None = None,
    clustering_method: str = "dbscan",
    n_prev: int = 1,
    predictor: bool | Callable = False,
    linking_method: str = 'nearest',
    allow_merges: bool = False,
    allow_splits: bool = False,
    stability_threshold: int = 10,
    remove_small_clusters: bool = False,
    min_size_for_split: int = 1,
    reg: float = 1,
    reg_m: float = 10,
    cost_threshold: float = 0,
    dims: str = "TXY",
    downsample: int = 1,
    n_jobs: int = 1,
    show_progress: bool = True,
    **kwargs,
) -> np.ndarray | tuple[np.ndarray, LineageTracker]:
    """Function to track events in an image using specified linking and clustering methods.

    Arguments:
        X (np.ndarray): The input array containing the images to track.
        eps (float): Distance for clustering. Default is 1.
        eps_prev (float | None): Maximum distance for linking previous clusters, if None, eps is used. Default is None.
        min_clustersize (int): Minimum cluster size. Default is 1.
        min_samples (int | None): The number of samples (or total weight) in a neighbourhood for a
            point to be considered as a core point. This includes the point itself.
            Only used if clusteringMethod is 'hdbscan'. If None, minSamples =  minClsz.
        clustering_method (str): The method used for clustering, one of [dbscan, hdbscan]. Default is "dbscan".
        n_prev (int): Number of previous frames to consider. Default is 1.
        predictor (bool | Callable): Whether or not to use a predictor. Default is False.
            True uses the default predictor. A callable can be passed to use a custom predictor.
            See default predictor method for details.
        linking_method (str): The method used for linking. Default is 'nearest'.
        allow_merges (bool): Whether or not to allow merges. Default is False.
        allow_splits (bool): Whether or not to allow splits. Default is False.
        stability_threshold (int): The number of frames required for a stable merge or split. Default is 10.
        remove_small_clusters (bool): Whether or not to remove small clusters. Default is False.
        min_size_for_split (int): Minimum size for a split. Default is 1.
        reg (float): Entropy regularization parameter for unbalanced OT algorithm (only for transportation linking).
        reg_m (float): Marginal relaxation parameter for unbalanced OT (only for transportation linking).
        cost_threshold (float): Threshold for filtering low-probability matches (only for transportation linking).
        dims (str): String of dimensions in order, such as. Default is "TXY". Possible values are "T", "X", "Y", "Z".
        downsample (int): Factor by which to downsample the image. Default is 1.
        n_jobs (int): Number of jobs to run in parallel. Default is 1.
        show_progress (bool): Whether or not to show progress bar. Default is True.
        **kwargs (Any): Additional keyword arguments. Includes deprecated parameters for backwards compatibility.
            - epsPrev: Deprecated parameter for eps_prev. Use eps_prev instead.
            - minClSz: Deprecated parameter for min_clustersize. Use min_clustersize instead.
            - minSamples: Deprecated parameter for min_samples. Use min_samples instead.
            - clusteringMethod: Deprecated parameter for clustering_method. Use clustering_method instead.
            - linkingMethod: Deprecated parameter for linking_method. Use linking_method instead.
            - nPrev: Deprecated parameter for n_prev. Use n_prev instead.
            - nJobs: Deprecated parameter for n_jobs. Use n_jobs instead.
            - showProgress: Deprecated parameter for show_progress. Use show_progress instead.

    Returns:
        np.ndarray: Array of images with tracked events.
    """
    map_params = {
        'epsPrev': 'eps_prev',
        'minClSz': 'min_clustersize',
        'minSamples': 'min_samples',
        'clusteringMethod': 'clustering_method',
        'linkingMethod': 'linking_method',
        'nPrev': 'n_prev',
        'nJobs': 'n_jobs',
        'showProgress': 'show_progress',
    }

    # check for allowed kwargs
    for key in kwargs:
        if key not in map_params.keys():
            raise ValueError(f'Invalid keyword argument {key}')

    # Handle deprecated parameters
    kwargs = handle_deprecated_params(map_params, **kwargs)

    # Assign parameters
    eps_prev = kwargs.get('eps_prev', eps_prev)
    min_clustersize = kwargs.get('min_clustersize', min_clustersize)
    min_samples = kwargs.get('min_samples', min_samples)
    clustering_method = kwargs.get('clustering_method', clustering_method)
    linking_method = kwargs.get('linking_method', linking_method)
    n_prev = kwargs.get('n_prev', n_prev)
    n_jobs = kwargs.get('n_jobs', n_jobs)

    # Determine the dimensionality
    spatial_dims = set("XYZ")
    D = len([d for d in dims if d in spatial_dims])

    # Adjust parameters based on dimensionality
    adjusted_epsPrev = eps_prev / downsample if eps_prev is not None else None
    adjusted_minClSz = int(min_clustersize / (downsample**D))
    adjusted_minSamples = int(min_samples / (downsample**D)) if min_samples is not None else None

    linker = Linker(
        eps=eps / downsample,
        eps_prev=adjusted_epsPrev,
        min_clustersize=adjusted_minClSz,
        min_samples=adjusted_minSamples,
        clustering_method=clustering_method,
        linking_method=linking_method,
        n_prev=n_prev,
        predictor=predictor,
        reg=reg,
        reg_m=reg_m,
        cost_threshold=cost_threshold,
        n_jobs=n_jobs,
        allow_merges=allow_merges,
        allow_splits=allow_splits,
        stability_threshold=stability_threshold,
        remove_small_clusters=remove_small_clusters,
        min_size_for_split=min_size_for_split,
    )
    tracker = ImageTracker(linker, downsample=downsample)
    # find indices of T in dims
    T_index = dims.upper().index("T")
    out = np.zeros_like(X, dtype=np.uint16)

    for i in tqdm(range(X.shape[T_index]), disable=not show_progress):
        out[i] = tracker.track_iteration(X[i])

    if any([allow_merges, allow_splits]):
        return out, linker.lineage_tracker

    return out


class detectCollev:
    """Class to detect collective events.

    Attributes:
        input_data (Union[pd.DataFrame, np.ndarray]): The input data to track.
        eps (float): Maximum distance for clustering, default is 1.
        epsPrev (Union[float, None]): Maximum distance for linking previous clusters, if None, eps is used.
            Default is None.
        minClSz (int): Minimum cluster size. Default is 3.
        nPrev (int): Number of previous frames to consider. Default is 1.
        posCols (list): List of column names for the position columns. Default is ["x"].
        frame_column (str): Name of the column containing the frame number. Default is 'time'.
        id_column (Union[str, None]): Name of the column containing the id. Default is None.
        bin_meas_column (Union[str, None]): Name of the column containing the binary measurement. Default is 'meas'.
        clid_column (str): Name of the column containing the cluster id. Default is 'clTrackID'.
        dims (str): String of dimensions in order, such as. Default is "TXY". Possible values are "T", "X", "Y", "Z".
        method (str): The method used for clustering, one of [dbscan, hdbscan]. Default is "dbscan".
        min_samples (int | None): The number of samples (or total weight) in a neighbourhood for a
                point to be considered as a core point. This includes the point itself.
                Only used if clusteringMethod is 'hdbscan'. If None, minSamples =  minClsz.
        linkingMethod (str): The method used for linking. Default is 'nearest'.
        n_jobs (int): Number of jobs to run in parallel. Default is 1.
        predictor (bool | Callable): Whether or not to use a predictor. Default is False.
            True uses the default predictor. A callable can be passed to use a custom predictor.
            See default predictor method for details.
        show_progress (bool): Whether or not to show progress bar. Default is True.
    """

    def __init__(
        self,
        input_data: Union[pd.DataFrame, np.ndarray],
        eps: float = 1,
        epsPrev: Union[float, None] = None,
        minClSz: int = 1,
        nPrev: int = 1,
        posCols: list = ["x"],
        frame_column: str = 'time',
        id_column: Union[str, None] = None,
        bin_meas_column: Union[str, None] = 'meas',
        clid_column: str = 'clTrackID',
        dims: str = "TXY",
        method: str = "dbscan",
        min_samples: int | None = None,
        linkingMethod='nearest',
        n_jobs: int = 1,
        predictor: bool | Callable = False,
        show_progress: bool = True,
    ) -> None:
        """Constructs class with input parameters.

        Arguments:
            input_data (DataFrame): Input data to be processed. Must contain a binarized measurement column.
            eps (float): The maximum distance between two samples for one to be considered as in
                the neighbourhood of the other.
                This is not a maximum bound on the distances of points within a cluster.
            epsPrev (float | None): Frame to frame distance, value is used to connect
                collective events across multiple frames.If "None", same value as eps is used.
            minClSz (int): Minimum size for a cluster to be identified as a collective event.
            nPrev (int): Number of previous frames the tracking
                algorithm looks back to connect collective events.
            posCols (list): List of position columns contained in the data.
                Must at least contain one.
            frame_column (str): Indicating the frame column in input_data.
            id_column (str | None): Indicating the track id/id column in input_data, optional.
            bin_meas_column (str): Indicating the bin_meas_column in input_data or None.
            clid_column (str): Indicating the column name containing the ids of collective events.
            dims (str): String of dimensions in order, used if input_data is a numpy array. Default is "TXY".
                Possible values are "T", "X", "Y", "Z".
            method (str): The method used for clustering, one of [dbscan, hdbscan]. Default is "dbscan".
            min_samples (int | None): The number of samples (or total weight) in a neighbourhood for a
                point to be considered as a core point. This includes the point itself.
                Only used if clusteringMethod is 'hdbscan'. If None, minSamples =  minClsz.
            linkingMethod (str): The method used for linking. Default is 'nearest'.
            n_jobs (int): Number of paralell workers to spawn, -1 uses all available cpus.
            predictor (bool | Callable): Whether or not to use a predictor. Default is False.
                True uses the default predictor. A callable can be passed to use a custom predictor.
                See default predictor method for details.
            show_progress (bool): Whether or not to show progress bar. Default is True.
        """
        self.input_data = input_data
        self.eps = eps
        self.epsPrev = epsPrev
        self.minClSz = minClSz
        self.nPrev = nPrev
        self.posCols = posCols
        self.frame_column = frame_column
        self.id_column = id_column
        self.bin_meas_column = bin_meas_column
        self.clid_column = clid_column
        self.dims = dims
        self.method = method
        self.linkingMethod = linkingMethod
        self.min_samples = min_samples
        self.predictor = predictor
        self.n_jobs = n_jobs
        self.show_progress = show_progress
        warnings.warn(
            "This class is deprecated and will be removed a future release, use the track_events_dataframe or track_events_image functions directly.",  # noqa: E501
            DeprecationWarning,
        )

    def run(self, copy: bool = True) -> pd.DataFrame:
        """Runs the collective event detection algorithm.

        Arguments:
            copy (bool): Whether or not to copy the input data. Default is True.

        Returns:
            DataFrame: Input data with added collective event ids.
        """
        if isinstance(self.input_data, pd.DataFrame):
            if copy:
                self.input_data = self.input_data.copy()
            return track_events_dataframe(
                X=self.input_data,
                position_columns=self.posCols,
                frame_column=self.frame_column,
                id_column=self.id_column,
                binarized_measurement_column=self.bin_meas_column,
                clid_column=self.clid_column,
                eps=self.eps,
                eps_prev=self.epsPrev,
                min_clustersize=self.minClSz,
                min_samples=self.min_samples,
                clustering_method=self.method,
                linking_method=self.linkingMethod,
                n_prev=self.nPrev,
                predictor=self.predictor,
                n_jobs=self.n_jobs,
                show_progress=self.show_progress,
            )
        elif isinstance(self.input_data, np.ndarray):
            if copy:
                self.input_data = np.copy(self.input_data)
            return track_events_image(
                X=self.input_data,
                eps=self.eps,
                eps_prev=self.epsPrev,
                min_clustersize=self.minClSz,
                min_samples=self.min_samples,
                clustering_method=self.method,
                n_prev=self.nPrev,
                predictor=self.predictor,
                linking_method=self.linkingMethod,
                dims=self.dims,
                n_jobs=self.n_jobs,
                show_progress=self.show_progress,
            )


def _nearest_neighbour_eps(
    X: np.ndarray,
    nbr_nearest_neighbours: int = 1,
):
    kdB = KDTree(data=X)
    nearest_neighbours, indices = kdB.query(X, k=nbr_nearest_neighbours)
    return nearest_neighbours[:, 1:]


def estimate_eps(
    data: pd.DataFrame,
    method: str = 'kneepoint',
    position_columns: list[str] = ['x,y'],
    frame_column: str = 't',
    n_neighbors: int = 5,
    plot: bool = True,
    plt_size: tuple[int, int] = (5, 5),
    max_samples=50_000,
    **kwargs: dict,
):
    """Estimates eps parameter in DBSCAN.

    Estimates the eps parameter for the DBSCAN clustering method, as used by ARCOS,
    by calculating the nearest neighbour distances for each point in the data.
    N_neighbours should be chosen to match the minimum point size in DBSCAN
    or the minimum clustersize in detect_events respectively.
    The method argument determines how the eps parameter is estimated.
    'kneepoint' estimates the knee of the nearest neighbour distribution.
    'mean' and 'median' return (by default) 1.5 times
    the mean or median of the nearest neighbour distances respectively.

    Arguments:
        data (pd.DataFrame): DataFrame containing the data.
        method (str, optional): Method to use for estimating eps. Defaults to 'kneepoint'.
            Can be one of ['kneepoint', 'mean', 'median'].'kneepoint' estimates the knee of the nearest neighbour
            distribution to to estimate eps. 'mean' and 'median' use the 1.5 times the mean or median of the
            nearest neighbour distances respectively.
        position_columns (list[str]): List of column names containing the position data.
        frame_column (str, optional): Name of the column containing the frame number. Defaults to 't'.
        n_neighbors (int, optional): Number of nearest neighbours to consider. Defaults to 5.
        plot (bool, optional): Whether to plot the results. Defaults to True.
        plt_size (tuple[int, int], optional): Size of the plot. Defaults to (5, 5).
        kwargs (Any): Keyword arguments for the method. Modify behaviour of respecitve method.
            For kneepoint: [S online, curve, direction, interp_method,polynomial_degree; For mean: [mean_multiplier]
            For median [median_multiplier]

    Returns:
        Eps (float): eps parameter for DBSCAN.
    """
    method_option = ['kneepoint', 'mean', 'median']

    if method not in method_option:
        raise ValueError(f"Method must be one of {method_option}")

    allowedtypes: dict[str, str] = {
        'kneepoint': 'kneepoint_values',
        'mean': 'mean_values',
        'median': 'median_values',
    }

    kwdefaults: dict[str, Any] = {
        'S': 1,
        'online': True,
        'curve': 'convex',
        'direction': 'increasing',
        'interp_method': 'polynomial',
        'mean_multiplier': 1.5,
        'median_multiplier': 1.5,
        'polynomial_degree': 7,
    }

    kwtypes: dict[str, Any] = {
        'S': int,
        'online': bool,
        'curve': str,
        'direction': str,
        'interp_method': str,
        'polynomial_degree': int,
        'mean_multiplier': (float, int),
        'median_multiplier': (float, int),
        'pos_cols': list,
        'frame_col': str,
    }

    allowedkwargs: dict[str, list[str]] = {
        'kneepoint_values': ['S', 'online', 'curve', 'interp_method', 'direction', 'polynomial_degree'],
        'mean_values': ['mean_multiplier'],
        'median_values': ['median_multiplier'],
    }

    map_deprecated_parameters = {
        'pos_cols': 'position_columns',
        'frame_col': 'frame_column',
    }

    for key in kwargs:
        if key not in allowedkwargs[allowedtypes[method]] and key not in map_deprecated_parameters:
            raise ValueError(f'{key} keyword not in allowed keywords {allowedkwargs[allowedtypes[method]]}')
        if not isinstance(kwargs[key], kwtypes[key]):
            raise ValueError(f'{key} must be of type {kwtypes[key]}')

    # Set kwarg defaults
    for kw in allowedkwargs[allowedtypes[method]]:
        kwargs.setdefault(kw, kwdefaults[kw])

    kwargs = handle_deprecated_params(map_deprecated_parameters, **kwargs)

    # assign parameters
    position_columns = kwargs.get('position_columns', position_columns)  # type: ignore
    frame_column = kwargs.get('frame_column', frame_column)  # type: ignore

    # remove deprecated parameters
    for key in map_deprecated_parameters:
        if key in kwargs:
            del kwargs[key]

    subset = [frame_column] + position_columns
    for i in subset:
        if i not in data.columns:
            raise ValueError(f"Column {i} not in data")

    subset = [frame_column] + position_columns
    data_np = data[subset].to_numpy(dtype=np.float64)
    # sort by frame
    data_np = data_np[data_np[:, 0].argsort()]
    grouped_array = np.split(data_np[:, 1:], np.unique(data_np[:, 0], axis=0, return_index=True)[1][1:])
    # map nearest_neighbours to grouped_array
    distances = [_nearest_neighbour_eps(i, n_neighbors) for i in grouped_array if i.shape[0] >= n_neighbors]
    if not distances:
        distances_array = np.array([])
    else:
        distances_array = np.concatenate(distances)
    # flatten array
    distances_flat = distances_array.flatten()
    distances_flat = distances_flat[np.isfinite(distances_flat)]
    distances_flat_selection = np.random.choice(
        distances_flat, min(max_samples, distances_flat.shape[0]), replace=False
    )
    distances_sorted = np.sort(distances_flat_selection)
    if distances_sorted.shape[0] == 0:
        raise ValueError('No valid distances found, please check input data.')
    if method == 'kneepoint':
        k1 = KneeLocator(
            np.arange(0, distances_sorted.shape[0]),
            distances_sorted,
            S=kwargs['S'],
            online=kwargs['online'],
            curve=kwargs['curve'],
            interp_method=kwargs['interp_method'],
            direction=kwargs['direction'],
            polynomial_degree=kwargs['polynomial_degree'],
        )

        eps = distances_sorted[k1.knee]

    elif method == 'mean':
        eps = np.mean(distances_sorted) * kwargs['mean_multiplier']

    elif method == 'median':
        eps = np.median(distances_sorted) * kwargs['median_multiplier']

    if plot:
        fig, ax = plt.subplots(figsize=plt_size)
        ax.plot(distances_sorted)
        ax.axhline(eps, color='r', linestyle='--')
        ax.set_xlabel('Sorted Distance Index')
        ax.set_ylabel('Nearest Neighbour Distance')
        ax.set_title(f'Estimated eps: {eps:.4f}')
        plt.show()

    return eps
