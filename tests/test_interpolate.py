import pandas as pd
import pytest
from numpy import nan, int64
from pandas.testing import assert_frame_equal

from arcos4py.tools import interpolation


@pytest.fixture
def test_data() -> pd.DataFrame:
    d = {
        't': [i if i != 4 else nan for i in range(1, 8)],
        'y': [i for i in range(1, 8)],
        'id': [1 for i in range(7)],
    }
    df = pd.DataFrame(d)

    d2 = {
        't': [i for i in range(1, 8)],
        'y': [i for i in range(1, 8)],
        'id': [1 for i in range(7)],
    }
    df2 = pd.DataFrame(d2)
    return df, df2


def test_interpolate_middle(test_data: pd.DataFrame):
    df, df_out = test_data
    interpolated_t = interpolation(df['t']).interpolate()
    df['t'] = interpolated_t
    df['t'] = df['t'].astype(int64)
    assert_frame_equal(df, df_out)
