import numpy as np

import pylake


def test_depth_filter_upcast():
    depth = np.array([
        0, 0, 0,
        0.1, 0.2, 0.3, 0.4,
        0.3, 0.5, 0.6,
        0.5, 0.4
    ])

    filtered = pylake.depth_filter(
        depth,
        run_length=3
    )

    np.testing.assert_array_equal(
        filtered,
        [0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    )


def test_depth_filter_indices():
    depth = np.array([
        0,
        0.1,
        0.2,
        0.15,
        0.3,
        0.4,
        0.3
    ])

    idx = pylake.depth_filter(
        depth,
        run_length=2,
        index=True
    )

    np.testing.assert_array_equal(
        idx,
        [0, 1, 2, 4, 5]
    )


def test_depth_filter_nan():
    depth = np.array([
        np.nan,
        0,
        0.1,
        0.2,
        np.nan,
        0.3
    ])

    filtered = pylake.depth_filter(
        depth,
        run_length=2
    )

    np.testing.assert_array_equal(
        filtered,
        [0, 0.1, 0.2, 0.3]
    )
