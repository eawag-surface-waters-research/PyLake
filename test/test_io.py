import json
import zipfile

import numpy as np
import pytest
import xarray as xr

from pylake.io import datalakes_to_xarray, read_datalakes, read_rsk, read_kor, read_tob


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


def test_datalakes_depth_order():
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
        [4, 2, 1],
    )

    np.testing.assert_array_equal(
        ds.temperature.values[0],
        [10, 15, 20],
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

    assert ds.temp.dims == ("depth", "time")
    assert ds.temp.shape == (3, 2)

    np.testing.assert_array_equal(
        ds.temp.values[:, 0],
        [20, 15, 10],
    )

    np.testing.assert_array_equal(
        ds.surfacetemp.values,
        [20, 19],
    )

    np.testing.assert_array_equal(
        ds.bottomtemp.values,
        [10, 9],
    )


def test_read_rsk(tmp_path):
    import sqlite3

    path = tmp_path / "data.rsk"

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE channels (
                channelID INTEGER,
                shortName TEXT,
                longName TEXT,
                units TEXT
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE data (
                tstamp INTEGER,
                channel01 REAL,
                channel02 REAL
            )
            """
        )

        connection.executemany(
            "INSERT INTO channels VALUES (?, ?, ?, ?)",
            [
                (1, "temp14", "Temperature", "°C"),
                (2, "pres24", "Pressure", "dbar"),
            ],
        )

        connection.executemany(
            "INSERT INTO data VALUES (?, ?, ?)",
            [
                (0, 20.0, 1.0),
                (1000, 19.0, 2.0),
            ],
        )

    ds = read_rsk(path)

    np.testing.assert_array_equal(
        ds.temp14.values,
        [20.0, 19.0],
    )

    np.testing.assert_array_equal(
        ds.pres24.values,
        [1.0, 2.0],
    )

    assert ds.temp14.attrs["units"] == "°C"
    assert ds.pres24.attrs["units"] == "dbar"
    assert ds.attrs["source"] == "RBR"


def test_read_kor(tmp_path):
    path = tmp_path / "data.csv"

    path.write_text(
        """sep=,
Kor MEASUREMENT DATA FILE EXPORT

FILE CREATED:,7/23/2025 12:19:32 PM

MEAN VALUE:
STANDARD DEVIATION:

SENSOR SERIAL NUMBER:
TIME (HH:MM:SS),DATE (MM/DD/YYYY),FAULT CODE,DEPTH M,TEMP °C,ODO MG/L
11:09:59 AM,7/23/2025,0,0.750,21.233,8.50
11:10:00 AM,7/23/2025,0,1.250,20.900,8.40
""",
        encoding="utf-16"
    )

    ds = read_kor(path)

    np.testing.assert_array_equal(
        ds["DEPTH M"].values,
        [0.750, 1.250],
    )

    np.testing.assert_array_equal(
        ds["TEMP °C"].values,
        [21.233, 20.900],
    )

    np.testing.assert_array_equal(
        ds["ODO MG/L"].values,
        [8.50, 8.40],
    )

    assert ds.attrs["source"] == "KOR"



def test_read_tob(tmp_path):
    path = tmp_path / "data.TOB"

    path.write_text(
        """header
; Datasets    Press      Temp       Cond     Chl_A     Turb       pH      sat    DO_mg      IntD      IntT
;            [ dbar]   [   °C]    [mS/cm]   [ µg/l]  [  FTU]  [    _]  [    %]  [ mg/l]   [ Date]   [ Time]
;
          1   -0.557  5.9526E+0  3.3507E-3   -2.239   -0.145    7.639   91.087   11.329 08/01/2021 14:22:51
          2   -0.573  5.9295E+0  3.4667E-3   -2.239   -0.145    7.640   91.371   11.372 08/01/2021 14:22:51
""",
        encoding="latin-1"
    )

    ds = read_tob(path)

    np.testing.assert_array_equal(
        ds.pressure.values,
        [-0.557, -0.573],
    )

    np.testing.assert_array_equal(
        ds.temperature.values,
        [5.9526, 5.9295],
    )

    np.testing.assert_array_equal(
        ds.oxygen_mg_l.values,
        [11.329, 11.372],
    )

    assert ds.attrs["source"] == "Sea & Sun"
