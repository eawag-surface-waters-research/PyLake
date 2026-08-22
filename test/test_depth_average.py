import numpy as np
import pylake


def test_depth_average():
    depth = np.array([1, 1, 2, 2, 3])
    values = np.array([10, 12, 20, 22, 30])

    d, v = pylake.depth_average(
        depth,
        values
    )

    np.testing.assert_array_equal(
        d,
        [1, 2, 3]
    )

    np.testing.assert_array_equal(
        v,
        [11, 21, 30]
    )


def test_depth_average_nan():
    depth = np.array([1, 1, 2, np.nan])
    values = np.array([10, np.nan, 20, 30])

    d, v = pylake.depth_average(
        depth,
        values
    )

    np.testing.assert_array_equal(
        d,
        [1, 2]
    )

    np.testing.assert_array_equal(
        v,
        [10, 20]
    )
