import numpy as np
import pandas as pd
import pytest

from arcos4py._arcos4py import ARCOS

# some tests for the main class. Not intended to be exhaustive. Since core functoinallity is tested independently.


def test_init():
    data = pd.DataFrame({'time': [1, 2, 3], 'id': [1, 1, 1], 'x': [5, 6, 7], 'meas': [10, 20, 30]})

    arcos = ARCOS(data)
    assert arcos.data.equals(data)
    assert arcos.position_columns == ['x']
    assert arcos.frame_column == 'time'
    assert arcos.obj_id_column == 'id'
    assert arcos.measurement_column == 'meas'
    assert arcos.clid_column == 'clTrackID'
    assert arcos.n_jobs == 1


def test_repr():
    data = pd.DataFrame({'time': [1, 2, 3], 'id': [1, 1, 1], 'x': [5, 6, 7], 'meas': [10, 20, 30]})

    arcos = ARCOS(data)
    assert repr(arcos) == repr(data)


def test_check_col_failure():
    data = pd.DataFrame(
        {
            'time': [1, 2, 3],
            'id': [1, 1, 1],
            'x': [5, 6, 7],
        }
    )

    with pytest.raises(ValueError):
        ARCOS(data, measurement_column='meas')


def test_interpolate_measurements():
    # Create test data with 'meas' values and NaNs
    data = pd.DataFrame(
        {'time': range(10), 'id': [1] * 10, 'x': [5] * 10, 'meas': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10]}
    )

    arcos = ARCOS(data)

    # Run interpolation method
    interpolated_data = arcos.interpolate_measurements()

    # Since the `interpolation` function is not defined in the class,
    # you would have to define what kind of interpolation you are doing.
    # Assuming a simple linear interpolation, the expected values would be:
    expected_measures = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    assert np.all(interpolated_data['meas'] == expected_measures), "Interpolation did not work as expected"


def test_clip_meas():
    # Create test data
    data = pd.DataFrame(
        {'time': range(10), 'id': [1] * 10, 'x': [5] * 10, 'meas': [1, 2, 3, 4, 5, 50, 70, 80, 90, 100]}
    )

    arcos = ARCOS(data)

    # Define quantile bounds
    clip_low = 10.9 / 100
    clip_high = 90.1 / 100

    # Get expected clipping values
    low_value = data['meas'].quantile(clip_low)
    high_value = data['meas'].quantile(clip_high)

    # Run clipping method with specified quantiles
    clipped_data = arcos.clip_measurements(clip_low=clip_low, clip_high=clip_high)

    # Check if the clipping worked as expected
    assert all(clipped_data['meas'] >= low_value), "Values below the lower bound were not clipped"
    assert all(clipped_data['meas'] <= high_value), "Values above the upper bound were not clipped"


def test_trackCollev():
    df_in = pd.read_csv('tests/testdata/1central_in.csv')
    df_true = pd.read_csv('tests/testdata/1central_res.csv')
    pos = ['x']
    ts = ARCOS(
        df_in,
        position_columns=pos,
        frame_column='time',
        obj_id_column='trackID',
        measurement_column='m',
        clid_column='clTrackID',
    )
    ts.binarized_measurement_column = 'm'
    out = ts.track_collective_events(eps=1, min_clustersize=1, n_prev=1)
    out = out.drop(columns=['m', 'x'])
    pd.testing.assert_frame_equal(out, df_true, check_dtype=False, check_like=True)
