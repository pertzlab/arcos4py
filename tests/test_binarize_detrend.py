import numpy as np
import pandas as pd
import pytest

from arcos4py.tools._binarize_detrend import binData, detrender


def test_detrender_init():
    det = detrender()
    assert det.smooth_k == 3
    assert det.bias_k == 51
    assert det.peak_threshold == 0.2
    assert det.polynomial_degree == 1
    assert det.bias_method == "runmed"
    assert det.n_jobs == 1

    with pytest.raises(ValueError):
        detrender(bias_method="invalid_method")


def test_detrender_runmed():
    x = np.array([1, 2, 3, 4, 5])
    det = detrender()
    result = det._detrend_runnmed(x, filter_size=3, endrule_mode="nearest")
    assert result.size == x.size


def test_binData_init():
    binarizer = binData()
    assert binarizer.smooth_k == 3
    assert binarizer.bias_k == 51
    assert binarizer.peak_threshold == 0.2
    assert binarizer.binarization_threshold == 0.1
    assert binarizer.polynomial_degree == 1
    assert binarizer.bias_method == "runmed"
    assert binarizer.n_jobs == 1


def test_binData_run_without_groupcol():
    binarizer = binData(bias_method="none")
    data = pd.DataFrame({'ERK_KTR': [1, 2, 3], 'Frame': [1, 2, 3]})
    result = binarizer.run(data, group_column=None, measurement_column='ERK_KTR', frame_column='Frame')
    assert 'ERK_KTR.resc' in result.columns
    assert 'ERK_KTR.bin' in result.columns


def test_binData_run_with_groupcol():
    binarizer = binData()
    data = pd.DataFrame({'trackID': [1, 1, 2], 'ERK_KTR': [1, 2, 3], 'Frame': [1, 2, 3]})
    result = binarizer.run(data, group_column='trackID', measurement_column='ERK_KTR', frame_column='Frame')
    assert 'ERK_KTR.resc' in result.columns
    assert 'ERK_KTR.bin' in result.columns


def test_median_detrending():
    df = pd.read_csv('tests/testdata/peak_data_linear_trend.csv')
    df_test = pd.read_csv('tests/testdata/peak_data_linear_trend_out.csv')
    binarizer = binData(smooth_k=1, bias_k=15, binarization_threshold=0.3, bias_method="runmed")
    df_out = binarizer.run(df, group_column='trackID', measurement_column='ERK_KTR', frame_column='Frame')
    pd.testing.assert_frame_equal(df_out, df_test, atol=0.1, check_dtype=False)


def test_polynomial_detrending():
    df = pd.read_csv('tests/testdata/peak_data_lm.csv')
    df_test = pd.read_csv('tests/testdata/peak_data_lm_out.csv')
    binarizer = binData(smooth_k=1, binarization_threshold=0.3, bias_method="lm")
    df_out = binarizer.run(df, group_column='trackID', measurement_column='ERK_KTR', frame_column='Frame')
    pd.testing.assert_frame_equal(df_out, df_test, atol=0.1, check_dtype=False)


def test_no_detrending():
    df = pd.read_csv('tests/testdata/peak_data.csv')
    df_test = pd.read_csv('tests/testdata/peak_data_out.csv')
    binarizer = binData(smooth_k=1, binarization_threshold=0.6, bias_method="none")
    df_out = binarizer.run(df, group_column='trackID', measurement_column='ERK_KTR', frame_column='Frame')
    pd.testing.assert_frame_equal(df_out, df_test, atol=0.1, check_dtype=False)


def test_no_detrending_without_groupcol():
    df = pd.read_csv('tests/testdata/peak_data.csv')
    df = df.query('trackID == 1')
    df_test = pd.read_csv('tests/testdata/peak_data_out.csv')
    df_test = df_test.query('trackID == 1')

    binarizer = binData(smooth_k=1, binarization_threshold=0.6, bias_method="none")
    df_out = binarizer.run(df, group_column=None, measurement_column='ERK_KTR', frame_column='Frame')
    pd.testing.assert_frame_equal(df_out, df_test, atol=0.1, check_dtype=False)


def test_detrending_with_groupcol():
    df = pd.read_csv('tests/testdata/peak_data.csv')
    binarizer = binData(smooth_k=1, binarization_threshold=0.6, bias_method="runmed")

    with pytest.warns(
        UserWarning, match="No detrending is performed, only rescaling. To run detrending, set colGroup."
    ):
        binarizer.run(df, group_column=None, measurement_column='ERK_KTR', frame_column='Frame')
