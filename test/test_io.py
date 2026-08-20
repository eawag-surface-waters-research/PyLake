import json
import zipfile

import numpy as np
import pytest
import xarray as xr

from pylake.io import datalakes_to_xarray, read_datalakes


def test_datalakes_to_xarray():
    data = {
        "x": [0, 600, 1200],
        "y": [1, 2, 4],
        "z": [
            [20, 19, 18],
            [15, 14, 13],
            [10, 9, 8],
        ],
        "y3": [1, 2, 3],
        "y4": [2, 3, 4],
    }

    ds = datalakes_to_xarray(data)

    assert ds.temperature.dims == ("time", "depth")
    assert ds.temperature.shape == (3, 3)

    np.testing.assert_array_equal(
        ds.depth.values,
        [1, 2, 4],
    )

    np.testing.assert_array_equal(
        ds.temperature.values[0],
        [20, 15, 10],
    )

    np.testing.assert_array_equal(
        ds.mixed_layer_depth.values,
        [1, 2, 3],
    )

    np.testing.assert_array_equal(
        ds.thermocline_depth.values,
        [2, 3, 4],
    )


def test_datalakes_descending_depth():
    data = {
        "x": [0, 600],
        "y": [4, 2, 1],
        "z": [
            [10, 9],
            [15, 14],
            [20, 19],
        ],
    }

    ds = datalakes_to_xarray(data)

    np.testing.assert_array_equal(
        ds.depth.values,
        [1, 2, 4],
    )

    np.testing.assert_array_equal(
        ds.temperature.values[0],
        [20, 15, 10],
    )


def test_datalakes_invalid_shape():
    data = {
        "x": [0, 600, 1200],
        "y": [1, 2],
        "z": [
            [20, 19],
            [18, 17],
        ],
    }

    with pytest.raises(ValueError):
        datalakes_to_xarray(data)


def test_datalakes_duplicate_depth():
    data = {
        "x": [0, 600],
        "y": [1, 1, 2],
        "z": [
            [20, 19],
            [18, 17],
            [10, 9],
        ],
    }

    with pytest.raises(ValueError):
        datalakes_to_xarray(data)


def test_read_datalakes_zip(tmp_path):
    data1 = {
        "x": [0, 600],
        "y": [1, 2, 4],
        "z": [
            [20, 19],
            [15, 14],
            [10, 9],
        ],
    }

    data2 = {
        "x": [1200, 1800],
        "y": [1, 2, 4],
        "z": [
            [18, 17],
            [13, 12],
            [8, 7],
        ],
    }

    path = tmp_path / "data.zip"

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "data1.json",
            json.dumps(data1),
        )
        archive.writestr(
            "data2.json",
            json.dumps(data2),
        )

    ds = read_datalakes(path)

    assert ds.temperature.dims == ("time", "depth")
    assert ds.temperature.shape == (4, 3)

    np.testing.assert_array_equal(
        ds.depth.values,
        [1, 2, 4],
    )

    np.testing.assert_array_equal(
        ds.temperature.values[0],
        [20, 15, 10],
    )


def test_read_datalakes_netcdf(tmp_path):
    path = tmp_path / "data.nc"

    data = xr.Dataset(
        {
            "temp": (
                ("depth", "time"),
                [
                    [20, 19],
                    [15, 14],
                    [10, 9],
                ],
            ),
            "surfacetemp": (
                "time",
                [20, 19],
            ),
            "bottomtemp": (
                "time",
                [10, 9],
            ),
        },
        coords={
            "time": np.array(
                [
                    "2026-01-01T00:00:00",
                    "2026-01-01T00:10:00",
                ],
                dtype="datetime64[ns]",
            ),
            "depth": [1, 2, 4],
        },
    )

    data.to_netcdf(
        path,
        engine="h5netcdf",
    )

    ds = read_datalakes(path)

    assert ds.temperature.dims == ("time", "depth")
    assert ds.temperature.shape == (2, 3)

    np.testing.assert_array_equal(
        ds.temperature.values[0],
        [20, 15, 10],
    )

    np.testing.assert_array_equal(
        ds.surface_temperature.values,
        [20, 19],
    )

    np.testing.assert_array_equal(
        ds.bottom_temperature.values,
        [10, 9],
    )
