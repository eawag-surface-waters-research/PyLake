import numpy as np

import pylake


def test_layer_average_constant():
    depth = np.array([0, 1, 2, 3])
    values = np.array([10, 10, 10, 10])

    bthD = np.array([0, 1, 2, 3])
    bthA = np.array([100, 80, 40, 0])

    result = pylake.layer_average(
        0,
        3,
        values,
        depth,
        bthA,
        bthD
    )

    np.testing.assert_allclose(
        result,
        10
    )


def test_layer_density():
    Temp = np.array([
        25.2,
        25.1,
        24.1,
        22.0,
        19.8,
        15.3,
        12.0,
        11.1
    ])

    depth = np.array([
        0, 1, 2, 3,
        4, 5, 6, 7
    ])

    bthA = np.array([
        10000,
        8900,
        5000,
        3500,
        2000,
        1000,
        300,
        10
    ])

    bthD = np.array([
        0, 1, 2, 3,
        4, 5, 6, 7
    ])

    result = pylake.layer_density(
        2,
        6,
        Temp,
        depth,
        bthA,
        bthD
    )

    assert np.isfinite(result)

    assert (
        990
        < result
        < 1005
    )


def test_layer_density_bounds():
    Temp = np.array([20, 15, 10])
    depth = np.array([0, 5, 10])

    bthD = np.array([0, 5, 10])
    bthA = np.array([1000, 500, 0])

    try:
        pylake.layer_density(
            8,
            2,
            Temp,
            depth,
            bthA,
            bthD
        )

    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError"
    )
