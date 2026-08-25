import numpy as np

import pylake


def test_ustar_positive():
    result = pylake.ustar(
        5.0,
        10,
        998.0
    )

    assert np.isfinite(result)
    assert result > 0


def test_ustar_zero_wind():
    result = pylake.ustar(
        0.0,
        10,
        998.0
    )

    np.testing.assert_allclose(
        result,
        0
    )


def test_ustar_vector():
    wind = np.array([
        2.0,
        4.0,
        6.0
    ])

    density = np.array([
        998.0,
        998.0,
        998.0
    ])

    result = pylake.ustar(
        wind,
        10,
        density
    )

    assert result.shape == wind.shape
    assert np.all(result > 0)
    assert np.all(np.diff(result) > 0)


def test_ustar_height():
    u10 = pylake.ustar(
        5.0,
        10,
        998.0
    )

    u2 = pylake.ustar(
        5.0,
        2,
        998.0
    )

    assert np.isfinite(u2)
    assert u2 != u10
