import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def datalakes_to_xarray(data, time_unit="s"):
    if isinstance(data, (str, Path)):
        with open(data, "r", encoding="utf-8") as f:
            data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError("")

    if not all(k in data for k in ["x", "y", "z"]):
        raise ValueError("")

    time = np.asarray(data["x"])

    if np.issubdtype(time.dtype, np.number):
        time = pd.to_datetime(
            time,
            unit=time_unit,
            utc=True
        ).tz_localize(None).to_numpy()
    else:
        time = pd.to_datetime(
            time,
            utc=True
        ).tz_localize(None).to_numpy()

    depth = np.asarray(data["y"], dtype=float)
    Temp = np.asarray(data["z"], dtype=float)
    if Temp.ndim != 2:
        raise ValueError("")

    if Temp.shape == (len(depth), len(time)):
        Temp = Temp.T
    elif Temp.shape != (len(time), len(depth)):
        raise ValueError("")

    ds = xr.Dataset(
        {
            "temperature": (
                ("time", "depth"),
                Temp
            )
        },
        coords={
            "time": time,
            "depth": depth
        }
    )

    variables = {
        "y1": "surface_temperature",
        "y2": "bottom_temperature",
        "y3": "mixed_layer_depth",
        "y4": "thermocline_depth",
        "y5": "schmidt_stability",
        "y6": "heat_content"
    }

    for old_name, new_name in variables.items():
        if old_name in data:
            values = np.asarray(data[old_name], dtype=float)
            ds[new_name] = ("time", values)

    ds.attrs["source"] = "DataLakes"

    return ds

def read_datalakes(path, time_unit="s"):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".json":
        return datalakes_to_xarray(
            path,
            time_unit=time_unit
        )

    if path.suffix.lower() == ".nc":
        with xr.open_dataset(
            path,
            engine="h5netcdf"
        ) as data:
            ds = data.load()

        ds.attrs["source"] = "DataLakes"

        return ds

    if path.suffix.lower() != ".zip":
        raise ValueError("")

    datasets = []

    with zipfile.ZipFile(path) as archive:
        json_files = [
            f for f in archive.namelist()
            if f.lower().endswith(".json")
        ]

        nc_files = [
            f for f in archive.namelist()
            if f.lower().endswith(".nc")
        ]

        if json_files:
            for name in json_files:
                data = json.loads(
                    archive.read(name).decode("utf-8")
                )

                datasets.append(
                    datalakes_to_xarray(
                        data,
                        time_unit=time_unit
                    )
                )

        elif nc_files:
            with tempfile.TemporaryDirectory() as tmp:
                for name in nc_files:
                    target = Path(tmp) / Path(name).name

                    target.write_bytes(
                        archive.read(name)
                    )

                    datasets.append(
                        read_datalakes(target)
                    )

        else:
            raise ValueError("")

    ds = xr.concat(
        datasets,
        dim="time"
    )

    ds.attrs["source"] = "DataLakes"

    return ds

