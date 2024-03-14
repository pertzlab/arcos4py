import numpy as np
import pandas as pd
import pytest

from arcos4py.validation._resampling import (
    _get_activity_blocks,
    _get_xy_change,
    resample_data,
    shift_timepoints_per_trajectory,
    shuffle_activity_bocks_per_trajectory,
    shuffle_coordinates_per_timepoint,
    shuffle_timepoints,
    shuffle_tracks,
)


def test__get_xy_change():
    # Create test dataframe
    test_df = pd.DataFrame({'track_id': [1, 1, 1, 2, 2, 2], 'x': [1, 2, 3, 4, 6, 8], 'y': [2, 3, 4, 5, 7, 9]})
    true_out = np.array([[0, 0], [1, 1], [2, 2], [0, 0], [2, 2], [4, 4]], dtype=np.int64)

    object_id_name = 'track_id'
    posCols = ['x', 'y']
    pos_cols_np = test_df[posCols].to_numpy()
    factorized_oid, uniques = pd.factorize(test_df[object_id_name])

    # Test the function
    cumsum_group = _get_xy_change(pos_cols_np, factorized_oid)

    # Assert that the output is as expected
    assert cumsum_group.shape == (6, 2)
    np.testing.assert_equal(true_out, cumsum_group)


def test_shuffle_tracks():
    # Create test dataframe
    test_df = pd.DataFrame(
        {'t': [1, 2, 3, 1, 2, 3], 'track_id': [1, 1, 1, 2, 2, 2], 'x': [1, 2, 3, 4, 6, 8], 'y': [2, 3, 4, 5, 7, 9]}
    )

    # Test the function with a specific seed
    df_new = shuffle_tracks(test_df, object_id_column='track_id', frame_column='t', seed=42)

    # Assert that the output is as expected
    assert df_new['track_id'].to_list() == [1, 1, 1, 2, 2, 2]
    assert df_new['x'].to_list() != [1, 2, 3, 4, 5, 6]
    assert df_new['y'].to_list() != [2, 3, 4, 5, 6, 7]

    # Assert that x and y stay in the correct pairs but with relative values
    # according to the cummulative change of the original track
    assert df_new.loc[df_new['track_id'] == 1, ['x', 'y']].to_numpy().tolist() == [[4, 5], [5, 6], [6, 7]]
    assert df_new.loc[df_new['track_id'] == 2, ['x', 'y']].to_numpy().tolist() == [[1, 2], [3, 4], [5, 6]]


def test_shuffle_timepoints():
    # Create test dataframe
    test_df = pd.DataFrame(
        {'track_id': [1, 1, 1, 2, 2, 2], 'time': [1, 2, 3, 1, 2, 3], 'x': [1, 2, 3, 4, 5, 6], 'y': [2, 3, 4, 5, 6, 7]}
    )

    # Test the function with a specific seed
    df_new = shuffle_timepoints(test_df, seed=42)

    # Assert that the output is as expected
    assert df_new['track_id'].to_list() == [1, 1, 1, 2, 2, 2]
    assert df_new['time'].to_list() == [1, 2, 3, 1, 2, 3]
    assert df_new['x'].to_list() != [1, 2, 3, 4, 5, 6]
    assert df_new['y'].to_list() != [2, 3, 4, 5, 6, 7]


def test__get_activity_blocks():
    # Create test data
    test_data = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1])

    # Test the function
    activity_blocks = _get_activity_blocks(test_data)

    # Assert that the output is as expected
    assert len(activity_blocks) == 4
    np.testing.assert_array_equal(activity_blocks[0], np.array([0, 0]))
    np.testing.assert_array_equal(activity_blocks[1], np.array([1, 1, 1]))
    np.testing.assert_array_equal(activity_blocks[2], np.array([0, 0]))
    np.testing.assert_array_equal(activity_blocks[3], np.array([1, 1, 1]))


def test_shuffle_activity_blocks_per_trajectory():
    # Create test dataframe
    test_df = pd.DataFrame(
        {
            'track_id': [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
            'time': [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8],
            'activity': [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        }
    )

    # Sort the test dataframe by time
    test_df.sort_values(by=['time', 'track_id'], inplace=True)

    # Test the function with a specific seed
    df_new = shuffle_activity_bocks_per_trajectory(
        test_df, 'track_id', 'time', 'activity', seed=42, alternating_blocks=True
    )

    # Assert that the output is as expected
    df_t_id_1 = df_new[df_new['track_id'] == 1]
    np.testing.assert_array_equal(df_t_id_1['track_id'].values, [1, 1, 1, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(df_t_id_1['time'].values, [1, 2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_array_equal(
        df_t_id_1['activity'].values, [0, 1, 0, 1, 0, 1, 0, 0]
    )  # Note that this is using a seed so assert_array_equal is possible

    df_t_id_2 = df_new[df_new['track_id'] == 2]
    np.testing.assert_array_equal(df_t_id_2['track_id'].values, [2, 2, 2, 2, 2, 2, 2, 2])
    np.testing.assert_array_equal(df_t_id_2['time'].values, [1, 2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_array_equal(
        df_t_id_2['activity'].values, [0, 1, 0, 0, 1, 0, 1, 0]
    )  # Note that this is using a seed so assert_array_equal is possible


def test_shuffle_coordinates_per_timepoint():
    # Create test dataframe
    test_df = pd.DataFrame(
        {
            'track_id': [1, 1, 1, 2, 2, 2],
            'time': [1, 2, 3, 1, 2, 3],
            'x': [1, 2, 3, 4, 5, 6],
            'y': [7, 8, 9, 10, 11, 12],
        }
    )
    # Test the function with a specific seed
    df_new = shuffle_coordinates_per_timepoint(test_df, ['x', 'y'], 'time', seed=42)

    # Assert that the output is as expected
    np.testing.assert_array_equal(df_new['track_id'].values, [1, 1, 1, 2, 2, 2])
    np.testing.assert_array_equal(df_new['time'].values, [1, 2, 3, 1, 2, 3])
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(df_new['x'].values, [1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(df_new['y'].values, [7, 8, 9, 10, 11, 12])


def test_shift_timepoints_per_trajectory():
    # Create test dataframe
    test_df = pd.DataFrame(
        {
            'track_id': [1, 1, 1, 2, 2, 2],
            'time': [1, 2, 3, 1, 2, 3],
            'x': [1, 2, 3, 4, 5, 6],
            'y': [7, 8, 9, 10, 11, 12],
        }
    )
    # Test the function with a specific seed
    df_new = shift_timepoints_per_trajectory(test_df, 'track_id', 'time', seed=42)

    # Assert that the output is as expected
    np.testing.assert_array_equal(df_new['track_id'].values, [1, 1, 1, 2, 2, 2])
    np.testing.assert_array_equal(df_new['time'].values, [1, 2, 3, 1, 2, 3])
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(df_new['x'].values, [1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(df_new['y'].values, [7, 8, 9, 10, 11, 12])


def test_resample_data_input_valid():
    data = pd.DataFrame(
        {
            'time': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'track_id': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
            'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'y': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
        }
    )

    posCols = ['x', 'y']
    frame_column = 'time'
    id_column = 'track_id'
    method = 'shuffle_tracks'
    n = 5
    seed = 42

    # test input validation
    with pytest.raises(ValueError, match='n must be a positive integer'):
        resample_data(data, posCols, frame_column, id_column, method=method, n=-1, seed=seed, verbose=False)
    with pytest.raises(ValueError, match='seed must be a positive integer'):
        resample_data(data, posCols, frame_column, id_column, method=method, n=n, seed=-1, verbose=False)
    with pytest.raises(TypeError, match='frame_column must be a string'):
        resample_data(data, posCols, 1, id_column, method=method, n=n, seed=seed, verbose=False)
    with pytest.raises(TypeError, match='id_column must be a string'):
        resample_data(data, posCols, frame_column, 1, method=method, n=n, seed=seed, verbose=False)
    with pytest.raises(TypeError, match='posCols must be a list'):
        resample_data(data, 'x', frame_column, id_column, method=method, n=n, seed=seed, verbose=False)
    with pytest.raises(TypeError, match='method must be a string or list'):
        resample_data(data, posCols, frame_column, id_column, method=1, n=n, seed=seed, verbose=False)
    with pytest.raises(TypeError, match='n must be a positive integer'):
        resample_data(data, posCols, frame_column, id_column, method=method, n='1', seed=seed, verbose=False)
    with pytest.raises(TypeError, match='seed must be an integer'):
        resample_data(data, posCols, frame_column, id_column, method=method, n=n, seed='1', verbose=False)
    with pytest.raises(TypeError, match='verbose must be a boolean'):
        resample_data(data, posCols, frame_column, id_column, method=method, n=n, seed=seed, verbose='False')
    with pytest.raises(ValueError, match='posCols must contain at least one column'):
        resample_data(data, [], frame_column, id_column, method=method, n=n, seed=seed, verbose=False)
    with pytest.raises(ValueError, match='method must be one of'):
        resample_data(data, posCols, frame_column, id_column, method='invalid_method', n=n, seed=seed, verbose=False)
    with pytest.raises(ValueError, match='y_not not in df.columns'):
        resample_data(data, ['y_not', 'x'], frame_column, id_column, method=method, n=n, seed=seed, verbose=False)


def test_resample_data():
    data = pd.DataFrame(
        {
            'time': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'track_id': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
            'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'y': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
        }
    )

    posCols = ['x', 'y']
    frame_column = 'time'
    id_column = 'track_id'
    method = 'shuffle_tracks'
    n = 5
    seed = 42

    resampled_data = resample_data(data, posCols, frame_column, id_column, method=method, n=n, seed=seed, verbose=False)
    # check that the resampled data is different from the original data
    for i in range(n):
        assert not data.equals(resampled_data)
    # check that the number of unique track_id is the same
    assert len(data[id_column].unique()) == len(resampled_data[id_column].unique())
    # check that the number of unique time is the same
    assert len(data[frame_column].unique()) == len(resampled_data[frame_column].unique())

    # test that the resampled data is the same when using the same seed
    resampled_data_2 = resample_data(
        data, posCols, frame_column, id_column, method=method, n=n, seed=seed, verbose=False
    )
    assert resampled_data.equals(resampled_data_2)

    # test that the resampled data is different when using a different seed
    resampled_data_3 = resample_data(
        data, posCols, frame_column, id_column, method=method, n=n, seed=seed + 1, verbose=False
    )
    assert not resampled_data.equals(resampled_data_3)
