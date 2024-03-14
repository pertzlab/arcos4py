import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist

from arcos4py.tools import calculate_statistics, calculate_statistics_per_frame


@pytest.fixture
def test_data():
    data = pd.DataFrame(
        {
            'collid': [1, 1, 1, 2, 2, 2],
            'frame': [1, 2, 3, 1, 2, 3],
            'id': [1, 2, 3, 4, 5, 6],
            'x': [0, 1, 2, 3, 4, 5],
            'y': [0, 1, 2, 3, 4, 5],
        }
    )
    return data


def test_calculate_summary_statistics(test_data):
    data = test_data
    # Test with obj_id_column and posCol
    stats_df = calculate_statistics(data, 'frame', 'collid', 'id', ['x', 'y'])

    # Expected output
    expected_output = pd.DataFrame(
        {
            'collid': [1, 2],
            'duration': [3, 3],
            'first_timepoint': [1, 1],
            'last_timepoint': [3, 3],
            'total_size': [3, 3],
            'min_size': [1, 1],
            'max_size': [1, 1],
            'first_frame_centroid_x': [0, 3],
            'last_frame_centroid_x': [2, 5],
            'first_frame_centroid_y': [0, 3],
            'last_frame_centroid_y': [2, 5],
            'centroid_speed': [2 * np.sqrt(2) / 2, 2 * np.sqrt(2) / 2],
            'direction': [np.pi / 4, np.pi / 4],
            'first_frame_spatial_extent': [0, 0],
            'first_frame_convex_hull_area': [0, 0],
            'last_frame_spatial_extent': [0, 0],
            'last_frame_convex_hull_area': [0, 0],
            'size_variability': [0, 0],
        }
    )

    pd.testing.assert_frame_equal(stats_df, expected_output, check_dtype=False)

    # Test with obj_id_column only
    stats_df = calculate_statistics(data, 'frame', 'collid', 'id', None)

    # Expected output
    expected_output = pd.DataFrame(
        {
            'collid': [1, 2],
            'duration': [3, 3],
            'first_timepoint': [1, 1],
            'last_timepoint': [3, 3],
            'total_size': [3, 3],
            'min_size': [1, 1],
            'max_size': [1, 1],
            'size_variability': [0.0, 0.0],
        }
    )

    pd.testing.assert_frame_equal(stats_df, expected_output)

    # Test with posCol only
    stats_df = calculate_statistics(data, 'frame', 'collid', None, ['x', 'y'])

    # Expected output
    expected_output = pd.DataFrame(
        {
            'collid': [1, 2],
            'duration': [3, 3],
            'first_timepoint': [1, 1],
            'last_timepoint': [3, 3],
            'min_size': [1, 1],
            'max_size': [1, 1],
            'first_frame_centroid_x': [0.0, 3.0],
            'last_frame_centroid_x': [2, 5.0],
            'first_frame_centroid_y': [0, 3],
            'last_frame_centroid_y': [2, 5],
            'centroid_speed': [2 * np.sqrt(2) / 2, 2 * np.sqrt(2) / 2],
            'direction': [np.pi / 4, np.pi / 4],
            'first_frame_spatial_extent': [0.0, 0.0],
            'first_frame_convex_hull_area': [0.0, 0.0],
            'last_frame_spatial_extent': [0.0, 0.0],
            'last_frame_convex_hull_area': [0.0, 0.0],
        }
    )

    pd.testing.assert_frame_equal(stats_df, expected_output, check_dtype=False)


def test_calculate_statistics_per_frame():
    # Test 1: Verify the function handles missing columns correctly
    data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    with pytest.raises(ValueError):
        calculate_statistics_per_frame(data, 'nonexistent_column', 'B')

    # Test 2: Test basic functionality with minimal input
    data = pd.DataFrame({'frame_column': [1, 2, 3], 'collid_column': [1, 1, 2]})
    result = calculate_statistics_per_frame(data, 'frame_column', 'collid_column')
    assert 'collid_column' in result.columns
    assert 'frame_column' in result.columns
    assert 'size' in result.columns

    # Test 3: Verify the correct calculation of centroids
    data = pd.DataFrame(
        {
            'frame_column': [1, 1, 2],
            'collid_column': [1, 1, 2],
            'x': [0, 1, 2],
            'y': [0, 1, 1],
        }
    )
    result = calculate_statistics_per_frame(data, 'frame_column', 'collid_column', position_columns=['x', 'y'])
    assert 'centroid_x' in result.columns
    assert 'centroid_y' in result.columns
    assert result.loc[result['frame_column'] == 1, 'centroid_x'].values[0] == 0.5
    assert result.loc[result['frame_column'] == 1, 'centroid_y'].values[0] == 0.5

    # Test 4: Verify correct calculation of spatial extent and convex hull area
    result = calculate_statistics_per_frame(data, 'frame_column', 'collid_column', position_columns=['x', 'y'])
    assert 'spatial_extent' in result.columns
    assert 'convex_hull_area' in result.columns
    assert (
        result.loc[result['frame_column'] == 1, 'spatial_extent'].values[0]
        == pdist(data.loc[data['frame_column'] == 1, ['x', 'y']].values).max()
    )

    # Test 5: Verify the correct calculation of direction and speed for 2D coordinates
    data = pd.DataFrame(
        {
            'frame_column': [1, 2, 1, 2],
            'collid_column': [1, 1, 2, 2],
            'x': [0, 1, 2, 3],
            'y': [0, 1, 1, 2],
        }
    )
    result = calculate_statistics_per_frame(data, 'frame_column', 'collid_column', position_columns=['x', 'y'])
    assert 'direction' in result.columns
    assert 'centroid_speed' in result.columns
    assert result.loc[result['frame_column'] == 2, 'direction'].values[0] == pytest.approx(np.arctan2(1, 1), rel=1e-5)
    assert result.loc[result['frame_column'] == 2, 'centroid_speed'].values[0] == pytest.approx(
        np.linalg.norm([1, 1]) / 1, rel=1e-5
    )

    # Test 7: Verify that the function handles single-object frames correctly (e.g., spatial_extent should be 0)
    data = pd.DataFrame(
        {
            'frame_column': [1],
            'collid_column': [1],
            'x': [0],
            'y': [0],
        }
    )
    result = calculate_statistics_per_frame(data, 'frame_column', 'collid_column', position_columns=['x', 'y'])
    assert result.loc[result['frame_column'] == 1, 'spatial_extent'].values[0] == 0
    assert result.loc[result['frame_column'] == 1, 'convex_hull_area'].values[0] == 0

    # Test 8: Verify that the function handles frames with two objects correctly (convex_hull_area should be 0)
    data = pd.DataFrame(
        {
            'frame_column': [1, 1],
            'collid_column': [1, 1],
            'x': [0, 1],
            'y': [0, 1],
        }
    )
    result = calculate_statistics_per_frame(data, 'frame_column', 'collid_column', position_columns=['x', 'y'])
    assert result.loc[result['frame_column'] == 1, 'convex_hull_area'].values[0] == 0
