#!/usr/bin/env python

"""Tests for `arcos_py` package."""

import pandas as pd
import pytest
from numpy import int64
from pandas.testing import assert_frame_equal

from arcos4py import ARCOS
from arcos4py.tools import track_events_dataframe


@pytest.fixture
def no_bin_data():
    """
    pytest fixture to generate test data
    """
    data = [item for i in range(10) for item in list(range(1, 11))]
    m = [0 for i in range(100)]
    d = {'id': data, 'time': data, 'm': m, 'x': data}
    print(d)
    df = pd.DataFrame(d)
    return df


def test_empty_data(no_bin_data: pd.DataFrame):
    with pytest.raises(ValueError, match='Input is empty'):
        test_data = no_bin_data[no_bin_data['m'] > 0]
        pos = ['x']
        ts = ARCOS(
            test_data,
            position_columns=pos,
            frame_column='time',
            obj_id_column='id',
            measurement_column='m',
            clid_column='clTrackID',
        )
        ts.track_collective_events(eps=1, eps_prev=None, min_clustersize=1, n_prev=2)


def test_1_central_1_prev():
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
    assert_frame_equal(out, df_true, check_dtype=False, check_like=True)


def test_1_central_2_prev():
    df_in = pd.read_csv('tests/testdata/1central_in.csv')
    df_true = pd.read_csv('tests/testdata/1central2prev_res.csv')
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
    out = ts.track_collective_events(eps=1, eps_prev=1, min_clustersize=1, n_prev=2)
    out = out.drop(columns=['m', 'x'])
    assert_frame_equal(out, df_true, check_dtype=False, check_like=True)


def test_1_central_3D():
    df_in = pd.read_csv('tests/testdata/1central3D_in.csv')
    df_true = pd.read_csv('tests/testdata/1central3D_res.csv')
    pos = ['x', 'y', 'z']
    ts = ARCOS(
        df_in,
        position_columns=pos,
        frame_column='time',
        obj_id_column='trackID',
        measurement_column='m',
        clid_column='clTrackID',
    )
    ts.binarized_measurement_column = 'm'
    out = ts.track_collective_events(eps=1, eps_prev=1, min_clustersize=1, n_prev=1)
    out = out.drop(columns=['m', 'x', 'y', 'z'])
    assert_frame_equal(out, df_true, check_like=True)


def test_1_central_growing():
    df_in = pd.read_csv('tests/testdata/1centralGrowing_in.csv')
    df_true = pd.read_csv('tests/testdata/1centralGrowing_res.csv')
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
    out = ts.track_collective_events(eps=1, eps_prev=1, min_clustersize=1, n_prev=1)
    out = out.drop(columns=['m', 'x'])
    assert_frame_equal(out, df_true, check_like=True)


def test_2_central_growing():
    df_in = pd.read_csv('tests/testdata/2centralGrowing_in.csv')
    df_true = pd.read_csv('tests/testdata/2centralGrowing_res.csv')
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
    out = ts.track_collective_events(eps=1, eps_prev=1, min_clustersize=1, n_prev=1)
    out = out.drop(columns=['m', 'x'])
    assert_frame_equal(out, df_true, check_like=True)


def test_2_with_1_common_symmetric():
    df_in = pd.read_csv('tests/testdata/2with1commonSym_in.csv')
    df_true = pd.read_csv('tests/testdata/2with1commonSym_res.csv')
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
    out = ts.track_collective_events(eps=1.0, eps_prev=1, min_clustersize=1, n_prev=1)
    out = out.drop(columns=['m', 'x'])
    assert_frame_equal(out, df_true, check_like=True)


def test_2_with_1_common_asymmetric():
    df_in = pd.read_csv('tests/testdata/2with1commonAsym_in.csv')
    df_true = pd.read_csv('tests/testdata/2with1commonAsym_res.csv')
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
    out = ts.track_collective_events(eps=1.0, eps_prev=1, min_clustersize=1, n_prev=1)
    out = out.drop(columns=['m', 'x'])
    assert_frame_equal(out, df_true, check_like=True)


def test_3_spreading_1_prev():
    df_in = pd.read_csv('tests/testdata/3spreading_in.csv')
    df_true = pd.read_csv('tests/testdata/3spreading_res.csv')
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
    out = ts.track_collective_events(eps=1.0, eps_prev=1, min_clustersize=1, n_prev=1)
    out = out.drop(columns=['m', 'x'])
    assert_frame_equal(out, df_true, check_like=True)


def test_3_spreading_2_prev():
    df_in = pd.read_csv('tests/testdata/3spreading_in.csv')
    df_true = pd.read_csv('tests/testdata/3spreading2prev_res.csv')
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
    out = ts.track_collective_events(eps=1.0, eps_prev=1, min_clustersize=1, n_prev=2)
    out = out.drop(columns=['m', 'x'])
    assert_frame_equal(out, df_true, check_like=True)


def test_5_overlapping_1_prev():
    df_in = pd.read_csv('tests/testdata/5overlapping_in.csv')
    df_true = pd.read_csv('tests/testdata/5overlapping_res.csv')
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
    out = ts.track_collective_events(eps=1.0, eps_prev=1, min_clustersize=1, n_prev=1)
    out = out.drop(columns=['m', 'x'])
    assert_frame_equal(out, df_true, check_like=True)


def test_5_overlapping_2_prev():
    df_in = pd.read_csv('tests/testdata/5overlapping_in.csv')
    df_true = pd.read_csv('tests/testdata/5overlapping2prev_res.csv')
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
    out = ts.track_collective_events(eps=1.0, eps_prev=1, min_clustersize=1, n_prev=2)
    out = out.drop(columns=['m', 'x'])
    assert_frame_equal(out, df_true, check_like=True)


def test_6_overlapping():
    df_in = pd.read_csv('tests/testdata/6overlapping_in.csv')
    df_true = pd.read_csv('tests/testdata/6overlapping_res.csv')
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
    out = ts.track_collective_events(eps=1.0, eps_prev=1, min_clustersize=1, n_prev=1)
    out = out.drop(columns=['m', 'x'])
    out['trackID'] = out['trackID'].astype(int64)
    assert_frame_equal(out, df_true, check_like=True)


def test_split_from_single():
    df_in = pd.read_csv('tests/testdata/1objSplit_in.csv')
    df_true = pd.read_csv('tests/testdata/1objSplit_res.csv')
    pos = ['pos']
    ts = ARCOS(
        df_in, position_columns=pos, frame_column='t', obj_id_column='id', measurement_column=None, clid_column='collid'
    )
    out = ts.track_collective_events(eps=1.0, eps_prev=1, min_clustersize=1, n_prev=1)
    out = out.drop(columns=['pos'])
    assert_frame_equal(out, df_true, check_like=True)


def test_split_from_2_objects():
    df_in = pd.read_csv('tests/testdata/2objSplit_in.csv')
    df_true = pd.read_csv('tests/testdata/2objSplit_res.csv')
    pos = ['pos']
    ts = ARCOS(
        df_in, position_columns=pos, frame_column='t', obj_id_column='id', measurement_column=None, clid_column='collid'
    )
    out = ts.track_collective_events(eps=1.0, eps_prev=1, min_clustersize=1, n_prev=1)
    out = out.drop(columns=['pos'])
    assert_frame_equal(out, df_true, check_like=True)


def test_4_colliding_with_allow_merges():
    """Test colliding event detection on a simple image."""
    test_df = pd.read_csv('tests/testdata/4obj_merge_allowed.csv')
    true_df = pd.read_csv('tests/testdata/4obj_merge_allowed_res.csv')
    # Sort the test data to ensure consistent order
    test_df = test_df.sort_values(by=['T', 'X', 'Y', 'track_id']).reset_index(drop=True)

    tracked_df, _ = track_events_dataframe(
        test_df,
        eps=2,
        eps_prev=2,
        min_clustersize=4,
        n_prev=1,
        position_columns=['X', 'Y'],
        frame_column='T',
        id_column='track_id',
        allow_merges=True,
        stability_threshold=1,
        allow_splits=True,
    )

    # Sort by relevant columns
    tracked_df = tracked_df.sort_values(by=['T', 'X', 'Y', 'track_id']).reset_index(drop=True)
    true_df = true_df.sort_values(by=['T', 'X', 'Y', 'track_id']).reset_index(drop=True)

    assert_frame_equal(tracked_df, true_df, check_dtype=False)


def test_cross_2_objects():
    df_in = pd.read_csv('tests/testdata/2objCross_in.csv')
    df_true = pd.read_csv('tests/testdata/2objCross_res.csv')
    pos = ['pos']
    ts = ARCOS(
        df_in, position_columns=pos, frame_column='t', obj_id_column='id', measurement_column=None, clid_column='collid'
    )
    out = ts.track_collective_events(eps=1.0, eps_prev=1, min_clustersize=1, n_prev=1)
    out = out.drop(columns=['pos'])
    assert_frame_equal(out, df_true, check_like=True)


def test_merge_split_2_objects_with_common():
    df_in = pd.read_csv('tests/testdata/2objMergeSplitCommon_in.csv')
    df_true = pd.read_csv('tests/testdata/2objMergeSplitCommon_res.csv')
    pos = ['pos']
    ts = ARCOS(
        df_in, position_columns=pos, frame_column='t', obj_id_column='id', measurement_column=None, clid_column='collid'
    )
    out = ts.track_collective_events(eps=1.0, eps_prev=1, min_clustersize=1, n_prev=1)
    out = out.drop(columns=['pos'])
    assert_frame_equal(out, df_true, check_like=True)


## if algorithm behaves differently (like in R) a different output is produced regarding collective events
def test_merge_split_2_objects_crossing():
    df_in = pd.read_csv('tests/testdata/2objMergeSplitCross_in.csv')
    df_true = pd.read_csv('tests/testdata/2objMergeSplitCross_res.csv')
    pos = ['pos']
    ts = ARCOS(
        df_in, position_columns=pos, frame_column='t', obj_id_column='id', measurement_column=None, clid_column='collid'
    )
    out = ts.track_collective_events(eps=1.0, eps_prev=1, min_clustersize=1, n_prev=1)
    out = out.drop(columns=['pos'])
    assert_frame_equal(out, df_true, check_like=True)


## if algorithm behaves differently (like in R) a different output is produced regarding collective events
def test_merge_and_split_2_objects_near():
    df_in = pd.read_csv('tests/testdata/2objMergeSplitNear_in.csv')
    df_true = pd.read_csv('tests/testdata/2objMergeSplitNear_res.csv')
    pos = ['pos']
    ts = ARCOS(
        df_in, position_columns=pos, frame_column='t', obj_id_column='id', measurement_column=None, clid_column='collid'
    )
    out = ts.track_collective_events(eps=1.0, eps_prev=1, min_clustersize=1, n_prev=1)
    out = out.drop(columns=['pos'])
    assert_frame_equal(out, df_true, check_like=True)


def test_4_objects_in_2_events():
    df_in = pd.read_csv('tests/testdata/4obj2events_in.csv')
    df_true = pd.read_csv('tests/testdata/4obj2events_res.csv')
    pos = ['x']
    ts = ARCOS(
        df_in,
        position_columns=pos,
        frame_column='frame',
        obj_id_column='id',
        measurement_column=None,
        clid_column='collId',
    )
    out = ts.track_collective_events(eps=1.0, eps_prev=1, min_clustersize=1, n_prev=1)
    out = out.drop(columns=['x'])
    assert_frame_equal(out, df_true, check_like=True)


def test_repeat_detection():
    df_in = pd.read_csv('tests/testdata/2objMergeSplitCross_in.csv')
    df_true = pd.read_csv('tests/testdata/2objMergeSplitCross_res.csv')
    pos = ['pos']
    ts = ARCOS(
        df_in, position_columns=pos, frame_column='t', obj_id_column='id', measurement_column=None, clid_column='collid'
    )
    out = ts.track_collective_events(eps=1.0, eps_prev=1, min_clustersize=1, n_prev=1)
    out = out.drop(columns=pos)
    assert_frame_equal(out, df_true)
    out = ts.track_collective_events(eps=1.0, eps_prev=1, min_clustersize=1, n_prev=1)
    out = out.drop(columns=pos)
    assert_frame_equal(out, df_true, check_like=True)


def test_repeat_with_different_eps():
    df_in = pd.read_csv('tests/testdata/2objMergeSplitCross_in.csv')
    df_true = pd.read_csv('tests/testdata/2objMergeSplitCross_res.csv')
    pos = ['pos']
    ts = ARCOS(
        df_in, position_columns=pos, frame_column='t', obj_id_column='id', measurement_column=None, clid_column='collid'
    )
    out = ts.track_collective_events(eps=0.01, eps_prev=1, min_clustersize=1, n_prev=1)
    out = out.drop(columns=pos)
    assert_frame_equal(out, df_true)
    out = ts.track_collective_events(eps=1.0, eps_prev=1, min_clustersize=1, n_prev=1)
    out = out.drop(columns=pos)
    assert_frame_equal(out, df_true, check_like=True)


def test_hdbscan_clustering():
    df_in = pd.read_csv('tests/testdata/arcos_2_synthetic_clusters.csv')
    df_true = pd.read_csv('tests/testdata/arcos_2_synthetic_clusters_hdbscan_true.csv')
    pos = ['x', 'y']
    ts = ARCOS(
        df_in,
        position_columns=pos,
        frame_column='t',
        obj_id_column='id',
        measurement_column='m',
        clid_column='collid',
    )
    ts.bin_measurements(binarization_threshold=0.5, bias_method='none')
    out = ts.track_collective_events(eps=0, eps_prev=2, min_clustersize=2, clustering_method='hdbscan')
    assert_frame_equal(out, df_true, check_like=True, check_dtype=False)
