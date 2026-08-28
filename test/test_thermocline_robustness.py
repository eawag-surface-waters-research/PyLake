import numpy as np
import pytest

import pylake
from pylake.functions import find_nearest, find_peak_index, set_nan


DEPTH = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
TEMP = np.array([14.3, 14, 12.1, 10, 9.7, 9.5, 6, 5], dtype=float)


def test_thermocline_weighted_is_default():
    default_depth, default_index = pylake.thermocline(TEMP, DEPTH)
    weighted_depth, weighted_index = pylake.thermocline(
        TEMP,
        DEPTH,
        weighted=True,
    )

    assert default_depth == pytest.approx(weighted_depth)
    assert default_index == weighted_index


def test_thermocline_unweighted_uses_interval_midpoint():
    depth, index = pylake.thermocline(
        TEMP,
        DEPTH,
        weighted=False,
    )

    assert depth == pytest.approx(2.5)
    assert index == 1


def test_thermocline_three_measurements_uses_midpoint():
    depth, index = pylake.thermocline(
        [20, 10, 9],
        [1, 2, 3],
    )

    assert depth == pytest.approx(1.5)
    assert index == 0


def test_thermocline_uniform_profile_returns_nan():
    depth, _ = pylake.thermocline(
        [10, 10, 10, 10],
        [0, 1, 2, 3],
    )

    assert np.isnan(depth)


def test_thermocline_rejects_duplicate_depths():
    with pytest.warns(UserWarning, match="depths must be unique"):
        depth, index = pylake.thermocline(
            [20, 15, 12, 10],
            [0, 1, 1, 2],
        )

    assert np.isnan(depth)
    assert np.isnan(index)


def test_thermocline_handles_multiple_profiles():
    temperatures = np.vstack([TEMP, TEMP - 1])
    depths, indices = pylake.thermocline(
        temperatures,
        DEPTH,
        time=["first", "second"],
    )

    assert depths.dims == ("time",)
    assert indices.dims == ("time",)
    assert np.all(np.isfinite(depths))
    np.testing.assert_array_equal(indices, [2, 2])


def test_thermocline_uses_local_intervals_for_irregular_depths():
    temperatures = np.array([21, 20.8, 20.4, 18, 12, 10, 9], dtype=float)
    depths = np.array([0, 0.4, 1.3, 2.7, 4.8, 7.5, 10], dtype=float)

    depth, index = pylake.thermocline(
        temperatures,
        depths,
        weighted=True,
    )

    density = pylake.water_density(temperatures, 0.2)
    gradient = np.diff(density) / np.diff(depths)
    interval = int(np.argmax(gradient))
    s_down = -(
        depths[interval + 1] - depths[interval]
    ) / (
        gradient[interval + 1] - gradient[interval]
    )
    s_up = (
        depths[interval] - depths[interval - 1]
    ) / (
        gradient[interval] - gradient[interval - 1]
    )
    expected = (
        depths[interval + 1] * s_down
        + depths[interval] * s_up
    ) / (s_down + s_up)

    assert depth == pytest.approx(expected)
    assert index == 3


def test_seasonal_thermocline_honors_unweighted_option():
    depth, _ = pylake.seasonal_thermocline(
        TEMP,
        DEPTH,
        seasonal_smoothed=False,
        weighted=False,
    )

    assert depth in {2.5, 6.5}


def test_find_peak_index_returns_deepest_peak():
    values = [0, 0.3, 0, 0.7, 0]

    assert find_peak_index(values, 0.1, 0) == 3


def test_find_peak_index_uses_fallback():
    values = [0, 0.01, 0]

    assert find_peak_index(values, 0.1, 2) == 2


def test_find_nearest_accepts_scalar_target():
    result = find_nearest(np.array([1, 2, 4]), 3.6)

    np.testing.assert_array_equal(result, [4])


def test_set_nan_accepts_scalar_and_vector_inputs():
    assert set_nan(1.0, 2.0) == 2.0
    assert np.isnan(set_nan(np.nan, 2.0))
    np.testing.assert_array_equal(
        set_nan([1, np.nan], [10, 20]),
        [10, np.nan],
    )
