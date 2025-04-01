import pandas as pd
import pytest

from arcos4py.tools import estimate_eps


@pytest.fixture
def test_data():
    data = {'t': [1, 1, 2, 2, 3, 3], 'x': [0, 1, 2, 3, 4, 5], 'y': [1, 2, 3, 4, 5, 6]}
    return pd.DataFrame(data)


def test_estimate_eps(test_data):
    # Test kneepoint method
    eps = estimate_eps(
        test_data,
        method='kneepoint',
        position_columns=['x', 'y'],
        frame_column='t',
        n_neighbors=1,
        S=1,
        online=True,
        curve='convex',
        interp_method='polynomial',
        direction='increasing',
        polynomial_degree=7,
        plot=False,
    )
    assert eps is not None

    # Test mean method
    eps = estimate_eps(
        test_data,
        method='mean',
        position_columns=['x', 'y'],
        frame_column='t',
        n_neighbors=1,
        mean_multiplier=1.5,
        plot=False,
    )
    assert eps is not None

    # Test median method
    eps = estimate_eps(
        test_data,
        method='median',
        position_columns=['x', 'y'],
        frame_column='t',
        n_neighbors=1,
        median_multiplier=1.5,
        plot=False,
    )
    assert eps is not None

    # Test invalid method
    with pytest.raises(ValueError):
        estimate_eps(
            test_data, method='invalid', position_columns=['x', 'y'], frame_column='t', n_neighbors=2, plot=False
        )

    # Test invalid column name
    with pytest.raises(ValueError):
        estimate_eps(test_data, position_columns=['x', 'y', 'z'], frame_column='t', n_neighbors=2, plot=False)

    # Test invalid kwarg type
    with pytest.raises(TypeError):
        estimate_eps(
            test_data,
            method='mean',
            position_columns=['x', 'y'],
            frame_column='t',
            n_neighbors=1,
            mean_multiplier='string',
            plot=False,
        )

    # Test invalid n_neighbors
    with pytest.raises(ValueError):
        eps = estimate_eps(
            test_data,
            method='kneepoint',
            position_columns=['x', 'y'],
            frame_column='t',
            n_neighbors=8,
            S=1,
            online=True,
            curve='convex',
            interp_method='polynomial',
            direction='increasing',
            polynomial_degree=7,
            plot=False,
        )
