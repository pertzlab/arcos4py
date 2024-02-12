"""Module to track and detect collective events.

Example:
    >>> from arcos4py.tools import track_events_image
    >>> ts = track_events_image(data)
"""

from __future__ import annotations

import functools
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp
from kneed import KneeLocator
from scipy.spatial.distance import cdist
from skimage.transform import rescale
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neighbors import KDTree
from tqdm import tqdm

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


def solve_transportation_problem(
    sources: np.ndarray,
    destinations: np.ndarray,
    supply_range: List[int],
    demand_range: List[int],
    max_distance: float,
) -> Tuple[Dict[Tuple[int, int], float], List[int], List[int]]:
    """Solves the transportation problem using the PuLP linear programming solver.

    Arguments:
        sources (np.ndarray): An array representing the source locations.
        destinations (np.ndarray): An array representing the destination locations.
        supply_range (List[int]): The range of the supply (minimum, maximum).
        demand_range (List[int]): The range of the demand (minimum, maximum).
        max_distance (float): The maximum allowable distance between a source and a destination.

    Returns:
        Tuple containing the assignments, supply, and demand as calculated by the linear programming solver.
    """
    # Calculate the costs (distances)
    costs = cdist(sources, destinations, metric='euclidean')

    # Set costs that exceed the max distance to a high value
    costs[costs > max_distance] = 1e6  # you can adjust this value as needed

    # Create the problem
    problem = pulp.LpProblem("TransportationProblem", pulp.LpMinimize)

    # Create decision variables for the transportation amounts
    x = pulp.LpVariable.dicts(
        "x", ((i, j) for i in range(len(sources)) for j in range(len(destinations))), lowBound=0, cat='Integer'
    )

    # Create decision variables for the supply and demand
    supply = [
        pulp.LpVariable(f"supply_{i}", lowBound=supply_range[0], upBound=supply_range[1], cat='Integer')
        for i in range(len(sources))
    ]
    demand = [
        pulp.LpVariable(f"demand_{j}", lowBound=demand_range[0], upBound=demand_range[1], cat='Integer')
        for j in range(len(destinations))
    ]

    # Objective function
    problem += pulp.lpSum(costs[i, j] * x[i, j] for i in range(len(sources)) for j in range(len(destinations)))

    # Supply constraints
    for i in range(len(sources)):
        problem += pulp.lpSum(x[i, j] for j in range(len(destinations))) <= supply[i]

    # Demand constraints
    for j in range(len(destinations)):
        problem += pulp.lpSum(x[i, j] for i in range(len(sources))) == demand[j]

    # Solve the problem
    problem.solve()

    # Get the assignments
    assignments = {
        (i, j): pulp.value(x[i, j])
        for i in range(len(sources))
        for j in range(len(destinations))
        if pulp.value(x[i, j]) > 0
    }

    # # Filter out assignments that exceed the maximum distance
    assignments = {(i, j): np.linalg.norm(sources[i] - destinations[j]) for (i, j) in assignments}

    # Return the results
    return (
        assignments,
        [pulp.value(supply[i]) for i in range(len(sources))],
        [pulp.value(demand[j]) for j in range(len(destinations))],
    )


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


def transportation_linking(
    cluster_labels: np.ndarray,
    cluster_coordinates: np.ndarray,
    memory_cluster_labels: np.ndarray,
    memory_coordinates: np.ndarray,
    epsPrev: float,
    max_cluster_label: int,
    supply_range: List[int],
    demand_range: List[int],
) -> Tuple[np.ndarray, int]:
    """Transportation linking of clusters across frames.

    Arguments:
        cluster_labels (np.ndarray): The cluster labels for the current frame.
        cluster_coordinates (np.ndarray): The cluster coordinates for the current frame.
        memory_cluster_labels (np.ndarray): The cluster labels for previous frames.
        memory_coordinates (np.ndarray): The coordinates for previous frames.
        epsPrev (float): Frame-to-frame distance, used to connect clusters across frames.
        max_cluster_label (int): The maximum label for clusters.
        supply_range (List[int]): The range of the supply (minimum, maximum).
        demand_range (List[int]): The range of the demand (minimum, maximum).

    Returns:
        Tuple containing the updated cluster labels and the maximum cluster label.
    """
    sources = memory_coordinates
    destinations = cluster_coordinates
    assignments, supply, demand = solve_transportation_problem(
        sources, destinations, supply_range, demand_range, epsPrev
    )
    cluster_labels = np.zeros(cluster_coordinates.shape[0])
    for (i, j), dist in assignments.items():
        if dist <= epsPrev:
            cluster_labels[j] = memory_cluster_labels[i]
        else:
            cluster_labels[j] = -1
    if any(cluster_labels == -1):
        max_cluster_label += 1
        cluster_labels[cluster_labels == -1] = max_cluster_label
    return cluster_labels, max_cluster_label


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

    def update(self, new_coordinates, new_cluster_ids):
        """Updates the coordinates and previous cluster IDs.

        Arguments:
            new_coordinates (np.ndarray): The new coordinates.
            new_cluster_ids (np.ndarray): The new cluster IDs.
        """
        self.remove_timepoint()
        self.add_timepoint(new_coordinates, new_cluster_ids)

    def add_timepoint(self, new_coordinates, new_cluster_ids):
        """Appends new coordinates and cluster IDs to the memory.

        Arguments:
            new_coordinates (np.ndarray): The new coordinates.
            new_cluster_ids (np.ndarray): The new cluster IDs.
        """
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


class Linker:
    """Linker class for linking collective events across multiple frames.

    Attributes:
        event_ids (np.ndarray): Array to store event IDs, for each coordinate in the current frame.

    Methods:
        link(input_coordinates): Links clusters from the previous frame to the current frame.
    """

    def __init__(
        self,
        eps: float = 1,
        epsPrev: float | None = None,
        minClSz: int = 1,
        minSamples: int | None = None,
        clusteringMethod: str | Callable = "dbscan",
        linkingMethod: str = "nearest",
        predictor: bool | Callable = True,
        nPrev: int = 1,
        nJobs: int = 1,
    ):
        """Initializes the Linker object.

        Arguments:
            eps (float): The maximum distance between two samples for one to be considered as in
                the neighbourhood of the other.
            epsPrev (float | None): Frame to frame distance, value is used to connect
                collective events across multiple frames. If "None", same value as eps is used.
            minClSz (int): The minimum size for a cluster to be identified as a collective event.
            minSamples (int | None): The number of samples (or total weight) in a neighbourhood for a
                point to be considered as a core point. This includes the point itself.
                Only used if clusteringMethod is 'hdbscan'. If None, minSamples =  minClsz.
            clusteringMethod (str | Callable): The clustering method to be used. One of ['dbscan', 'hdbscan']
                or a callable that takes a 2d array of coordinates and returns a list of cluster labels.
                Arguments `eps`, `minClSz` and `minSamples` are ignored if a callable is passed.
            linkingMethod (str): The linking method to be used.
            predictor (bool | Callable): The predictor method to be used.
            nPrev (int): Number of previous frames the tracking
                algorithm looks back to connect collective events.
            nJobs (int): Number of jobs to run in parallel (only for clustering algorithm).
        """
        self._predictor: Predictor | None  # for mypy
        self._memory = Memory(n_timepoints=nPrev)

        if callable(predictor):
            self._predictor = Predictor(predictor)
        elif predictor:
            self._predictor = Predictor.with_default_predictor()
        else:
            self._predictor = None

        self._nn_tree: KDTree | None = None
        if epsPrev is None:
            self._epsPrev = eps
        else:
            self._epsPrev = epsPrev
        self._n_jobs = nJobs
        self._validate_input(eps, epsPrev, minClSz, minSamples, clusteringMethod, nPrev, nJobs)

        self.event_ids = np.empty((0, 0), dtype=np.int64)
        self.max_prev_event_id = 0

        if hasattr(clusteringMethod, '__call__'):  # check if it's callable
            self.clustering_function = clusteringMethod
        else:
            if clusteringMethod == "dbscan":
                self.clustering_function = functools.partial(_dbscan, eps=eps, minClSz=minClSz)
            elif clusteringMethod == "hdbscan":
                self.clustering_function = functools.partial(
                    _hdbscan, eps=eps, minClSz=minClSz, min_samples=minSamples, cluster_selection_method='eom'
                )
            else:
                raise ValueError(
                    f'Clustering method must be either in {AVAILABLE_CLUSTERING_METHODS} or a callable with data as the only argument an argument'  # noqa E501
                )

        if hasattr(linkingMethod, '__call__'):  # check if it's callable
            self.linking_function = linkingMethod
        else:
            if linkingMethod == "nearest":
                self.linking_function = 'brute_force_linking'
            elif linkingMethod == "transportation":
                self.linking_function = 'transportation_linking'
            else:
                raise ValueError(
                    f'Linking method must be either in {AVAILABLE_LINKING_METHODS} or a callable'  # noqa E501
                )

    def _validate_input(self, eps, epsPrev, minClSz, minSamples, clusteringMethod, nPrev, nJobs):
        if not isinstance(eps, (int, float, str)):
            raise ValueError(f"eps must be a number or None, got {eps}")
        if not isinstance(epsPrev, (int, float, type(None))):
            raise ValueError(f"{epsPrev} must be a number or None, got {epsPrev}")
        if not isinstance(minSamples, (int, type(None))):
            raise ValueError(f"{minSamples} must be a number or None, got {minSamples}")
        for i in [minClSz, nPrev, nJobs]:
            if not isinstance(i, int):
                raise ValueError(f"{i} must be an int, got {i}")
        if not isinstance(clusteringMethod, str) and not callable(clusteringMethod):
            raise ValueError(f"clusteringMethod must be a string or a callable, got {clusteringMethod}")

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
                epsPrev=self._epsPrev,
                max_cluster_label=self._memory.max_prev_cluster_id,
            )
        elif self.linking_function == 'transportation_linking':
            linked_clusters, max_cluster_label = transportation_linking(
                cluster_labels=cluster,
                cluster_coordinates=cluster_coordinates,
                memory_cluster_labels=self._memory.all_cluster_ids,
                memory_coordinates=self._memory.all_coordinates,
                epsPrev=self._epsPrev,
                max_cluster_label=self._memory.max_prev_cluster_id,
                supply_range=[1, 10],
                demand_range=[1, 10],
            )
        else:
            raise ValueError(f'Linking method must be (for now) in {AVAILABLE_LINKING_METHODS}')

        self._memory.max_prev_cluster_id = max_cluster_label

        return linked_clusters

    def _update_tree(self, coords):
        self._nn_tree = KDTree(coords)

    # @profile
    def link(self, input_coordinates: np.ndarray) -> None:
        """Links clusters from the previous frame to the current frame.

        Arguments:
            input_coordinates (np.ndarray): The coordinates of the current frame.

        Returns:
            None, modifies internal state with new linked clusters. New event ids are stored in self.event_ids.
        """
        cluster_ids, coordinates, nanrows = self._clustering(input_coordinates)
        # check if first frame
        if not len(self._memory.prev_cluster_ids):
            linked_cluster_ids = self._update_id_empty(cluster_ids)
        # check if anything was detected in current or previous frame
        elif cluster_ids.size == 0 or self._memory.all_cluster_ids.size == 0:
            linked_cluster_ids = self._update_id_empty(cluster_ids)
        else:
            linked_cluster_ids = self._update_id(cluster_ids, coordinates)

        # update memory with current frame and fit predictor if necessary
        self._memory.add_timepoint(new_coordinates=coordinates, new_cluster_ids=linked_cluster_ids)
        if self._predictor is not None and len(self._memory.coordinates) > 1:
            self._predictor.fit(coordinates=self._memory.coordinates, cluster_ids=self._memory.prev_cluster_ids)
        self._memory.remove_timepoint()

        event_ids = np.full_like(nanrows, -1, dtype=np.int64)
        event_ids[~nanrows] = linked_cluster_ids
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
        coordinates_column: list[str],
        frame_column: str,
        id_column: str | None = None,
        bin_meas_column: str | None = None,
        collid_column: str = 'clTrackID',
    ):
        """Initializes the DataFrameTracker object.

        Arguments:
            linker (Linker): The Linker object used for linking events.
            coordinates_column (list[str]): List of strings representing the coordinate columns.
            frame_column (str): String representing the frame/timepoint column in the dataframe.
            id_column (str | None): String representing the ID column, or None if not present. Defaults to None.
            bin_meas_column (str | None): String representing the binary measurement column, or None if not present.
                Defaults to None.
            collid_column (str): String representing the collision track ID column. Defaults to 'clTrackID'.
        """
        super().__init__(linker)
        self._coordinates_column = coordinates_column
        self._frame_column = frame_column
        self._id_column = id_column
        self._bin_meas_column = bin_meas_column
        self._collid_column = collid_column
        self._validate_input(coordinates_column, frame_column, id_column, bin_meas_column, collid_column)

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
        pos_col: list[str],
    ) -> np.ndarray:
        """Select necessary input colums from input data and returns them as numpy arrays.

        Arguments:
            data (DataFrame): Containing necessary columns.
            frame_col (str): Name of the frame/timepoint column in the dataframe.
            pos_col (list): string representation of position columns in data.

        Returns:
            np.ndarray, np.ndarray: Filtered columns necessary for calculation.
        """
        pos_columns_np = data[pos_col].to_numpy()
        if pos_columns_np.ndim == 1:
            pos_columns_np = pos_columns_np[:, np.newaxis]
        return pos_columns_np

    def _sort_input(self, x: pd.DataFrame, frame_col: str, object_id_col: str | None) -> pd.DataFrame:
        """Sorts the input dataframe according to the frame column and track id column if available."""
        if object_id_col:
            x = x.sort_values([frame_col, object_id_col]).reset_index(drop=True)
        else:
            x = x.sort_values([frame_col]).reset_index(drop=True)
        return x

    def _filter_active(self, data: pd.DataFrame, bin_meas_col: Union[str, None]) -> pd.DataFrame:
        """Selects rows with binary value of greater than 0.

        Arguments:
            data (DataFrame): Dataframe containing necessary columns.
            bin_meas_col (str|None): Either name of the binary column or None if no such column exists.

        Returns:
            DataFrame: Filtered pandas DataFrame.
        """
        if bin_meas_col is not None:
            data = data[data[bin_meas_col] > 0]
        return data

    # @profile
    def track_iteration(self, x: pd.DataFrame) -> pd.DataFrame:
        """Tracks events in a single frame. Returns dataframe with event ids.

        Arguments:
            x (pd.DataFrame): Dataframe to track.

        Returns:
            pd.DataFrame: Dataframe with event ids.
        """
        x_filtered = self._filter_active(x, self._bin_meas_column)

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
        x_sorted = self._sort_input(x, frame_col=self._frame_column, object_id_col=self._id_column)

        for t in range(x_sorted[self._frame_column].max() + 1):
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

    def _filter_active(self, pos_data: np.ndarray, bin_meas_data: np.ndarray) -> np.ndarray:
        """Selects rows with binary value of greater than 0.

        Arguments:
            frame_data (np.ndarray): frame column as a numpy array.
            pos_data (np.ndarray): positions/coordinate columns as a numpy array.
            bin_meas_data (np.ndarray): binary measurement column as a numpy array.

        Returns:
            np.ndarray: Filtered numpy arrays.
        """
        if bin_meas_data is not None:
            active = np.argwhere(bin_meas_data > 0).flatten()
            pos_data = pos_data[active]
        return pos_data

    def _coordinates_to_image(self, x, pos_data, tracked_events):
        # create empty image
        out_img = np.zeros_like(x, dtype=np.uint16)
        if tracked_events.size == 0:
            return out_img
        tracked_events_mask = tracked_events > 0

        pos_data = pos_data[tracked_events_mask].astype(np.uint16)
        n_dims = pos_data.shape[1]

        # Raise an error if dimension is zero
        if n_dims == 0:
            raise ValueError("Dimension of input array not supported.")

        # Create an indexing tuple
        indices = tuple(pos_data[:, i] for i in range(n_dims))

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
    coordinates_column: List[str],
    frame_column: str,
    id_column: str | None,
    bin_meas_column: str | None = None,
    collid_column: str = "collid",
    eps: float = 1.0,
    epsPrev: float | None = None,
    minClSz: int = 3,
    minSamples: int | None = None,
    clusteringMethod: str = "dbscan",
    linkingMethod: str = 'nearest',
    nPrev: int = 1,
    predictor: bool | Callable = False,
    nJobs: int = 1,
    showProgress: bool = True,
) -> pd.DataFrame:
    """Function to track collective events in a dataframe.

    Arguments:
        X (pd.DataFrame): The input dataframe containing the data to track.
        coordinates_column (List[str]): The names of the columns representing coordinates.
        frame_column (str): The name of the column containing frame ids.
        id_column (str | None): The name of the column representing IDs. None if no such column.
        bin_meas_column (str | None): The name of the column representing binarized measurements,
            if None all measurements are used.
        collid_column (str): The name of the output column representing collective events, will be generated.
        eps (float): Maximum distance for clustering, default is 1.
        epsPrev (float | None): Maximum distance for linking previous clusters, if None, eps is used. Default is None.
        minClSz (int): Minimum cluster size. Default is 3.
        minSamples (int): The number of samples (or total weight) in a neighbourhood for a
            point to be considered as a core point. This includes the point itself.
            Only used if clusteringMethod is 'hdbscan'. If None, minSamples =  minClsz.
        clusteringMethod (str): The method used for clustering, one of [dbscan, hdbscan]. Default is "dbscan".
        linkingMethod (str): The method used for linking, one of ['nearest', 'transportsolver']. Default is 'nearest'.
        nPrev (int): Number of previous frames to consider. Default is 1.
        predictor (bool | Callable): Whether or not to use a predictor. Default is False.
            True uses the default predictor. A callable can be passed to use a custom predictor.
            See default predictor method for details.
        nJobs (int): Number of jobs to run in parallel. Default is 1.
        showProgress (bool): Whether or not to show progress bar. Default is True.

    Returns:
        pd.DataFrame: Dataframe with tracked events.
    """
    linker = Linker(
        eps=eps,
        epsPrev=epsPrev,
        minClSz=minClSz,
        minSamples=minSamples,
        clusteringMethod=clusteringMethod,
        linkingMethod=linkingMethod,
        nPrev=nPrev,
        predictor=predictor,
        nJobs=nJobs,
    )
    tracker = DataFrameTracker(
        linker=linker,
        coordinates_column=coordinates_column,
        frame_column=frame_column,
        id_column=id_column,
        bin_meas_column=bin_meas_column,
        collid_column=collid_column,
    )
    df_out = pd.concat(
        [timepoint for timepoint in tqdm(tracker.track(X), total=X[frame_column].nunique(), disable=not showProgress)]
    ).reset_index(drop=True)
    return df_out.query(f"{collid_column} != -1").reset_index(drop=True)


def track_events_image(
    X: np.ndarray,
    eps: float = 1,
    epsPrev: float | None = None,
    minClSz: int = 1,
    minSamples: int | None = None,
    clusteringMethod: str = "dbscan",
    nPrev: int = 1,
    predictor: bool | Callable = False,
    linkingMethod: str = 'nearest',
    dims: str = "TXY",
    downsample: int = 1,
    nJobs: int = 1,
    showProgress: bool = True,
) -> np.ndarray:
    """Function to track events in an image using specified linking and clustering methods.

    Arguments:
        X (np.ndarray): The input array containing the images to track.
        eps (float): Distance for clustering. Default is 1.
        epsPrev (float | None): Maximum distance for linking previous clusters, if None, eps is used. Default is None.
        minClSz (int): Minimum cluster size. Default is 1.
        minSamples (int | None): The number of samples (or total weight) in a neighbourhood for a
            point to be considered as a core point. This includes the point itself.
            Only used if clusteringMethod is 'hdbscan'. If None, minSamples =  minClsz.
        clusteringMethod (str): The method used for clustering, one of [dbscan, hdbscan]. Default is "dbscan".
        nPrev (int): Number of previous frames to consider. Default is 1.
        predictor (bool | Callable): Whether or not to use a predictor. Default is False.
            True uses the default predictor. A callable can be passed to use a custom predictor.
            See default predictor method for details.
        linkingMethod (str): The method used for linking. Default is 'nearest'.
        dims (str): String of dimensions in order, such as. Default is "TXY". Possible values are "T", "X", "Y", "Z".
        downsample (int): Factor by which to downsample the image. Default is 1.
        nJobs (int): Number of jobs to run in parallel. Default is 1.
        showProgress (bool): Whether or not to show progress bar. Default is True.

    Returns:
        np.ndarray: Array of images with tracked events.
    """
    # Determine the dimensionality
    spatial_dims = set("XYZ")
    D = len([d for d in dims if d in spatial_dims])

    # Adjust parameters based on dimensionality
    adjusted_epsPrev = epsPrev / downsample if epsPrev is not None else None
    adjusted_minClSz = int(minClSz / (downsample**D))
    adjusted_minSamples = int(minSamples / (downsample**D)) if minSamples is not None else None

    linker = Linker(
        eps=eps / downsample,
        epsPrev=adjusted_epsPrev,
        minClSz=adjusted_minClSz,
        minSamples=adjusted_minSamples,
        clusteringMethod=clusteringMethod,
        linkingMethod=linkingMethod,
        nPrev=nPrev,
        predictor=predictor,
        nJobs=nJobs,
    )
    tracker = ImageTracker(linker, downsample=downsample)
    # find indices of T in dims
    T_index = dims.upper().index("T")
    return np.stack(
        [timepoint for timepoint in tqdm(tracker.track(X, dims), total=X.shape[T_index], disable=not showProgress)],
        axis=T_index,
    )


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
                coordinates_column=self.posCols,
                frame_column=self.frame_column,
                id_column=self.id_column,
                bin_meas_column=self.bin_meas_column,
                collid_column=self.clid_column,
                eps=self.eps,
                epsPrev=self.epsPrev,
                minClSz=self.minClSz,
                minSamples=self.min_samples,
                clusteringMethod=self.method,
                linkingMethod=self.linkingMethod,
                nPrev=self.nPrev,
                predictor=self.predictor,
                nJobs=self.n_jobs,
                showProgress=self.show_progress,
            )
        elif isinstance(self.input_data, np.ndarray):
            if copy:
                self.input_data = np.copy(self.input_data)
            return track_events_image(
                X=self.input_data,
                eps=self.eps,
                epsPrev=self.epsPrev,
                minClSz=self.minClSz,
                minSamples=self.min_samples,
                clusteringMethod=self.method,
                nPrev=self.nPrev,
                predictor=self.predictor,
                linkingMethod=self.linkingMethod,
                dims=self.dims,
                nJobs=self.n_jobs,
                showProgress=self.show_progress,
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
    pos_cols: list[str] = ['x,y'],
    frame_col: str = 't',
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
        pos_cols (list[str]): List of column names containing the position data.
        frame_col (str, optional): Name of the column containing the frame number. Defaults to 't'.
        n_neighbors (int, optional): Number of nearest neighbours to consider. Defaults to 5.
        plot (bool, optional): Whether to plot the results. Defaults to True.
        plt_size (tuple[int, int], optional): Size of the plot. Defaults to (5, 5).
        kwargs: Keyword arguments for the method. Modify behaviour of respecitve method.
            For kneepoint: [S online, curve, direction, interp_method,polynomial_degree; For mean: [mean_multiplier]
            For median [median_multiplier]

    Returns:
        Eps (float): eps parameter for DBSCAN.
    """
    subset = [frame_col] + pos_cols
    for i in subset:
        if i not in data.columns:
            raise ValueError(f"Column {i} not in data")
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
    }

    allowedkwargs: dict[str, list[str]] = {
        'kneepoint_values': ['S', 'online', 'curve', 'interp_method', 'direction', 'polynomial_degree'],
        'mean_values': ['mean_multiplier'],
        'median_values': ['median_multiplier'],
    }

    for key in kwargs:
        if key not in allowedkwargs[allowedtypes[method]]:
            raise ValueError(f'{key} keyword not in allowed keywords {allowedkwargs[allowedtypes[method]]}')
        if not isinstance(kwargs[key], kwtypes[key]):
            raise ValueError(f'{key} must be of type {kwtypes[key]}')

    # Set kwarg defaults
    for kw in allowedkwargs[allowedtypes[method]]:
        kwargs.setdefault(kw, kwdefaults[kw])

    subset = [frame_col] + pos_cols
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
