"""Tools for resampling data."""

from __future__ import annotations

import warnings
from itertools import zip_longest
from typing import TYPE_CHECKING, Callable, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

if TYPE_CHECKING:
    import numpy.typing as npt

from ..tools._arcos4py_deprecation import handle_deprecated_params


def _np_diff(arr: list[np.ndarray], axis: int):
    return [np.cumsum(np.diff(np.insert(i, 0, i[0], axis=0), axis=0), axis=0) for i in arr]


def _get_xy_change(X: np.ndarray, object_ids: np.ndarray) -> tuple[pd.DataFrame, list[str]]:
    """Calculate xy change for each object."""
    # get xy change for each object
    # make sure that X and object_ids are numpy arrays of the same length
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(object_ids, np.ndarray):
        object_ids = np.array(object_ids)
    if X.shape[0] != object_ids.shape[0]:
        raise ValueError('X and object_ids must have the same length.')

    # check if object_ids are sorted
    if not np.all(object_ids[:-1] <= object_ids[1:]):
        raise ValueError('object_ids must be sorted.')

    # calculate xy change
    grouped_array = np.split(X, np.unique(object_ids, axis=0, return_index=True)[1][1:])
    cumsum_group = _np_diff(grouped_array, axis=0)
    return np.concatenate(cumsum_group)


def shuffle_tracks(
    df: pd.DataFrame, object_id_column: str = 'track_id', positoin_column: list = ['x', 'y'], frame_column='t', seed=42
) -> pd.DataFrame:
    """Resample tracks by switching the first timepoint\
        positions of two tracks and then propagating the cummulative difference."""
    df.sort_values([object_id_column, frame_column], inplace=True)  # needs to be sorted for _get_xy_change to work
    df_pos_cols_np = df[positoin_column].to_numpy()
    factorized_oid, _ = pd.factorize(df[object_id_column])

    # Set the random seed
    rng = np.random.default_rng(seed)
    # Get unique track IDs
    unique_track_ids = np.unique(factorized_oid)
    # Keep track of the track IDs that have been switched
    switched_tracks = set()
    # calculate relative coordinate change
    xy_change = _get_xy_change(df_pos_cols_np, factorized_oid)

    # Iterate over the unique track IDs

    for track_id in unique_track_ids:
        # Skip the track if it has already been switched
        if track_id in switched_tracks:
            continue
        switched_tracks.add(track_id)
        # Get a random track ID that has not been switched yet
        mask = np.isin(unique_track_ids, list(switched_tracks), invert=True)
        # Use the boolean mask to select the track IDs that meet the criteria
        available_track_ids = unique_track_ids[mask]

        if available_track_ids.size == 0:
            continue
        random_track_id = rng.choice(available_track_ids)
        # Get the rows with the current track ID
        track_rows = np.argwhere(factorized_oid == track_id)
        track_rows = track_rows.flatten()
        # Get the rows with the random track ID
        random_track_rows = np.argwhere(factorized_oid == random_track_id)
        random_track_rows = random_track_rows.flatten()

        # Switch the first timepoint positions of the two tracks
        start_pos_random = df_pos_cols_np[random_track_rows[0]].copy()
        start_pos_track = df_pos_cols_np[track_rows[0]].copy()

        df_pos_cols_np[track_rows] = start_pos_random
        df_pos_cols_np[random_track_rows] = start_pos_track

        # Add the relative position shift to the rest of the timepoints for the current track
        df_pos_cols_np[track_rows] += xy_change[track_rows]
        # Add the relative position shift to the rest of the timepoints for the random track
        df_pos_cols_np[random_track_rows] += xy_change[random_track_rows]
        switched_tracks.add(random_track_id)
    df_out = df.copy(deep=True)
    df_out[positoin_column] = df_pos_cols_np
    return df_out


def shuffle_timepoints(
    df: pd.DataFrame,
    objet_id_column: str = 'track_id',
    frame_column: str = 'time',
    seed=42,
) -> pd.DataFrame:
    """Resample data by shuffling data from timepoints on a per trajectory basis."""
    df_new = df.copy(deep=True)
    # Set the random seed
    rng = np.random.default_rng(seed)
    # Get unique track IDs
    track_ids = df_new[objet_id_column].factorize()[0]
    unique_track_ids = np.unique(track_ids)

    # get timepoints
    df_t_np = df_new[frame_column].to_numpy()
    # Iterate over the unique track IDs
    for track_id in unique_track_ids:
        # Get the rows with the current track ID
        track_rows = np.argwhere(track_ids == track_id)
        track_rows = track_rows.flatten()
        df_t_shuffle = df_t_np[track_rows]
        # Shuffle the timepoints
        rng.shuffle(df_t_shuffle)
        df_t_np[track_rows] = df_t_shuffle

    df_new[frame_column] = df_t_np
    df_new.sort_values(by=[objet_id_column, frame_column], inplace=True)
    return df_new


def _get_activity_blocks(data: np.ndarray) -> list[np.ndarray]:
    """Get the activity blocks of a binary activity column."""
    # Get the indices of the activity blocks
    activity_block_indices = np.where(np.diff(data) != 0)[0] + 1
    # Get the activity blocks
    activity_blocks = np.split(data, activity_block_indices)
    # Remove empty activity blocks
    activity_blocks = [block for block in activity_blocks if block.size > 0]
    return activity_blocks


def shuffle_activity_bocks_per_trajectory(
    df: pd.DataFrame,
    object_id_column: str,
    frame_column: str,
    measurement_column: str,
    seed=42,
    alternating_blocks=True,
) -> pd.DataFrame:
    """Resample data by shuffling the activity blocks of a binary activity column on a per trajectory basis."""
    df_new = df.copy(deep=True)
    # check if data in meas_column is binary
    if not np.array_equal(np.unique(df_new[measurement_column]), np.array([0, 1])):
        raise ValueError('Data in meas_column must be binary')

    # raise warning if 1 makes up more than 25% of the array
    if np.sum(df_new[measurement_column]) / df_new[measurement_column].size > 0.25:
        warnings.warn(
            'More than 25%% of the data in meas_column is 1.\
            This could impact the validity of this resampling approach.'
        )

    # Set the random seed
    rng = np.random.default_rng(seed)
    # Get unique track IDs
    track_ids = df_new[object_id_column].unique()
    # Iterate over the unique track IDs
    for track_id in track_ids:
        # Get the rows with the current track ID
        track_rows = df_new[df_new[object_id_column] == track_id].copy()
        # Get the activity blocks
        meas_array = track_rows[measurement_column].to_numpy()
        activity_blocks = _get_activity_blocks(meas_array)
        # if alternating blocks is True, group the activity blocks by its value,
        # then suffle them separately and alternate them

        if alternating_blocks:
            # check if the first block is 1 or 0
            if activity_blocks[0][0] == 1:
                first_block = 1
                second_block = 0
            else:
                first_block = 0
                second_block = 1
            # group the activity blocks by its value
            block_first = [block for block in activity_blocks if block[0] == first_block]
            block_second = [block for block in activity_blocks if block[0] == second_block]
            rng.shuffle(block_first)
            rng.shuffle(block_second)
            activity_blocks = []
            for b1, b0 in zip_longest(block_first, block_second):
                if b1 is not None:
                    activity_blocks.append(b1)
                if b0 is not None:
                    activity_blocks.append(b0)
        else:
            # Shuffle the activity blocks
            rng.shuffle(activity_blocks)
        # Set the shuffled activity blocks
        df_new.loc[track_rows.index, measurement_column] = np.concatenate(activity_blocks)
    df_new.sort_values(by=[frame_column, object_id_column], inplace=True)
    return df_new


def shuffle_coordinates_per_timepoint(
    df: pd.DataFrame,
    position_columns: list = ['x', 'y'],
    frame_column: str = 'time',
    seed=42,
) -> pd.DataFrame:
    """Resample data by shuffling the coordinates of a per timepoint basis."""
    df_new = df.copy(deep=True)
    # Set the random seed
    rng = np.random.default_rng(seed)
    # Get unique timepoints
    timepoints = df_new[frame_column].to_numpy()
    unique_timepoints = np.unique(timepoints)

    # get coordinates
    df_tp_col_np = df_new[position_columns].to_numpy()
    # Iterate over the unique track IDs
    for tp in unique_timepoints:
        # Get the rows with the current track ID
        tp_rows = np.argwhere(timepoints == tp)
        tp_rows = tp_rows.flatten()
        df_pos_shuffle = df_tp_col_np[tp_rows]
        # Shuffle the coordinates
        rng.shuffle(df_pos_shuffle)
        df_tp_col_np[tp_rows] = df_pos_shuffle

    df_new[position_columns] = df_tp_col_np
    return df_new


def shift_timepoints_per_trajectory(
    df: pd.DataFrame, object_id_column: str, frame_column: str, seed=42
) -> pd.DataFrame:
    """Resample data by shifting the timepoints a random ammount of a trajectory on a per trajectory basis."""
    df_new = df.copy(deep=True)
    # Set the random seed
    rng = np.random.default_rng(seed)
    # Get unique track IDs
    track_ids = df_new[object_id_column].factorize()[0]
    unique_track_ids = np.unique(track_ids)

    # get timepoints
    df_t_np = df_new[frame_column].to_numpy()
    # Iterate over the unique track IDs
    for track_id in unique_track_ids:
        # Get the rows with the current track ID
        track_rows = np.argwhere(track_ids == track_id)
        track_rows = track_rows.flatten()
        timepoints = df_t_np[track_rows]
        # Get the shift
        shift = rng.integers(-timepoints.size, timepoints.size)
        # Shift the timepoints
        timepoints = np.roll(timepoints, shift)
        df_t_np[track_rows] = timepoints

    df_new.sort_values(by=[object_id_column, frame_column], inplace=True)
    return df_new


def resample_data(  # noqa: C901
    data: pd.DataFrame,
    position_columns: list,
    frame_column: str,
    obj_id_column: str,
    measurement_column: Union[str, None] = None,
    method: Union[str, list[str]] = 'shuffle_tracks',
    n=100,
    seed=42,
    allow_duplicates=False,
    max_tries=100,
    show_progress=True,
    verbose=False,
    parallel_processing=True,
    **kwargs,
) -> pd.DataFrame:
    """Resamples data in order to perform bootstrapping analysis.

    Arguments:
        data (pd.Dataframe): The data to resample.
        position_columns (list): The columns to use for the position.
        frame_column (str): The column to use for the frame.
        obj_id_column (str): The column to use for the object ID.
        measurement_column (str, optional): The column to use for the measurement.
            Only needed for 'activity_blocks_shuffle'. Defaults to None.
        method (str, optional): The method to use for resampling. Defaults to 'shuffle_tracks'.
            Available methods are: "shuffle_tracks", 'shuffle_timepoints',
            'shift_timepoints', 'shuffle_binary_blocks', 'shuffle_coordinates_timepoint'
        n (int, optional): The number of resample iterations. Defaults to 100.
        seed (int, optional): The random seed. Defaults to 42.
        allow_duplicates (bool, optional): Whether to allow resampling to randomly generate the same data twice.
            Defaults to False.
        max_tries (int, optional): The maximum number of tries to try ot generate unique data
            when allow_duplicates is set to True. Defaults to 100.
        verbose (bool, optional): Whether to print progress. Defaults to False.
        parallel_processing (bool, optional): Whether to use parallel processing. Defaults to True.
        **kwargs (Any): Additional keyword arguments. Includes deprecated parameters.
            - posCols (list): Deprecated. Use position_columns instead.
            - id_column (str): Deprecated. Use obj_id_column instead.
            - meas_column (str): Deprecated. Use measurement_column instead.
            - paralell_processing (bool): Deprecated. Use parallel_processing instead.

    Returns:
        pd.DataFrame: The resampled data.
    """
    map_deprecated_params = {
        "posCols": "position_columns",
        "id_column": "obj_id_column",
        "meas_column": "measurement_column",
        "paralell_processing": "parallel_processing",
    }

    # check allowed kwargs
    allowed_kwargs = map_deprecated_params.keys()
    for key in kwargs:
        if key not in allowed_kwargs:
            raise ValueError(f"Got an unexpected keyword argument '{key}'")
    updated_kwargs = handle_deprecated_params(map_deprecated_params, **kwargs)

    position_columns = updated_kwargs.get("position_columns", position_columns)
    obj_id_column = updated_kwargs.get("obj_id_column", obj_id_column)
    measurement_column = updated_kwargs.get("measurement_column", measurement_column)
    parallel_processing = updated_kwargs.get("parallel_processing", parallel_processing)

    # validate the input
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be a pandas.DataFrame')
    if not isinstance(position_columns, list):
        raise TypeError('posCols must be a list')
    if not isinstance(frame_column, str):
        raise TypeError('frame_column must be a string')
    if not isinstance(obj_id_column, str):
        raise TypeError('id_column must be a string')
    if not isinstance(measurement_column, str) and measurement_column is not None:
        raise TypeError('meas_column must be a string or None')
    if not isinstance(method, str) and not isinstance(method, list):
        raise TypeError('method must be a string or list')
    if not isinstance(n, int):
        raise TypeError('n must be a positive integer')
    if not isinstance(seed, int):
        raise TypeError('seed must be an integer')
    if not isinstance(verbose, bool):
        raise TypeError('verbose must be a boolean')
    if not isinstance(parallel_processing, bool):
        raise TypeError('paralell_processing must be a boolean')

    if len(position_columns) < 1:
        raise ValueError('posCols must contain at least one column')
    if n < 1:
        raise ValueError('n must be a positive integer')
    if seed < 0:
        raise ValueError('seed must be a positive integer')

    method_dict: dict[str, Callable] = {
        'shuffle_tracks': shuffle_tracks,
        'shuffle_timepoints': shuffle_timepoints,
        'shift_timepoints': shift_timepoints_per_trajectory,
        'shuffle_binary_blocks': shuffle_activity_bocks_per_trajectory,
        'shuffle_coordinates_timepoint': shuffle_coordinates_per_timepoint,
    }

    function_args: dict[str, tuple] = {
        'shuffle_tracks': (obj_id_column, position_columns, frame_column),
        'shuffle_timepoints': (obj_id_column, frame_column),
        'shift_timepoints': (obj_id_column, frame_column),
        'shuffle_binary_blocks': (obj_id_column, frame_column, measurement_column),
        'shuffle_coordinates_timepoint': (position_columns, frame_column),
    }

    resampling_func_list = []

    # convert method to list if necessary
    if isinstance(method, str):
        methods = [method]
    else:
        methods = method

    # Check if the method is valid
    for method in methods:
        if method not in method_dict.keys():
            raise ValueError(f'method must be one of {method_dict.keys()}')
        if method == 'shuffle_binary_blocks' and measurement_column is None:
            raise ValueError('meas_column must be set for binary_blocks_shuffle')

    # Check if the columns are in the data
    if 'shuffle_binary_blocks' in methods:
        relevant_columns = position_columns + [frame_column, obj_id_column, measurement_column]
    else:
        relevant_columns = position_columns + [frame_column, obj_id_column]

    for i in relevant_columns:
        if i not in data.columns:
            raise ValueError(f'{i} not in df.columns')

    # check if there are any Nan in the columns selected
    na_cols = []
    for i in relevant_columns:
        if data[position_columns].isnull().values.any():
            na_cols.append(i)
    if na_cols:
        warnings.warn(f'NaN values in {na_cols}, default behaviour is to drop these rows')
        data.dropna(subset=na_cols, inplace=True)

    # Sort the data
    data.sort_values([obj_id_column, frame_column], inplace=True)

    rng = np.random.default_rng(seed)
    # create a list of random numbers between 0 and 1000000
    seed_list = rng.integers(1_000_000_000, size=n)
    df_out: list[pd.DataFrame] = []
    # shuffle xy position for each object
    if verbose:
        print(f'Resampling for each object {n} times')

    # create a list of functions to call
    for method in methods:
        resampling_func_list.append(method_dict[method])
    iter_range = range(1, n + 1)
    if parallel_processing:
        from joblib import Parallel, delayed

        # iterate over the number of resamples
        df_out = Parallel(n_jobs=-1)(
            delayed(_apply_resampling)(
                iter_number=i,
                data=data,
                methods=methods,
                resampling_func_list=resampling_func_list,
                seed_list=seed_list,
                function_args=function_args,
            )
            for i in tqdm(iter_range, disable=not show_progress)
        )

    else:
        # iterate over the number of resamples
        for i in tqdm(iter_range, disable=not show_progress):
            data_new = _apply_resampling(
                iter_number=i,
                data=data,
                methods=methods,
                resampling_func_list=resampling_func_list,
                seed_list=seed_list,
                function_args=function_args,
            )
            if not allow_duplicates:
                current_try = 0
                # make sure that data_new is not already in df_out,
                # but they are both dataframes, else redo the resampling
                while any(
                    data_new.loc[:, data_new.columns != 'iteration'].equals(i.loc[:, i.columns != 'iteration'])
                    for i in df_out
                ):
                    current_try += 1
                    data_new = _apply_resampling(
                        iter_number=i,
                        data=data,
                        methods=methods,
                        resampling_func_list=resampling_func_list,
                        seed_list=seed_list,
                        function_args=function_args,
                    )
                    if current_try > max_tries:
                        raise ValueError(
                            'Could not find a unique resampling after 100 tries, try increasing n or allow_duplicates'
                        )

            df_out.append(data_new)

    data_it0 = data.copy()
    data_it0['iteration'] = np.repeat(0, len(data_it0))
    df_out.insert(0, data_it0)
    return pd.concat(df_out)[data.columns.tolist() + ['iteration']]


def _apply_resampling(
    iter_number: int,
    data: pd.DataFrame,
    methods: list[str],
    resampling_func_list: list[Callable],
    seed_list: npt.NDArray[np.int64],
    function_args: dict[str, tuple],
) -> pd.DataFrame:
    """Resamples data in order to perform bootstrapping analysis.

    Arguments:
        iter_number (int): The iteration number
        data (pd.Dataframe): The data to resample.
        methods (list[str]): The methods to use for resampling.
        resampling_func_list list(Callable): The function to use for resampling.
        seed_list list(int): The random seed.
        function_args (dict[str, Callable]): The arguments for the resamping. function.

    Returns:
        pd.DataFrame: The resampled data.
    """
    data_new = data.copy()
    _seed = seed_list[iter_number - 1]

    # iterate over the resampling functions
    for resampling_func, method in zip(resampling_func_list, methods):
        data_new = resampling_func(data_new, *function_args[method], seed=_seed)

    data_new['iteration'] = np.repeat(iter_number, len(data_new))
    return data_new
