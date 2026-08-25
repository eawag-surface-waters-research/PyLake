import json
import sqlite3
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _detect_source(path):
    """Detect a supported data source from its extension or file signature."""
    path = Path(path)
    suffix = path.suffix.lower()

    suffix_sources = {
        ".json": "datalakes",
        ".nc": "datalakes",
        ".zip": "datalakes",
        ".rsk": "rsk",
        ".csv": "kor",
        ".tob": "tob",
    }

    if suffix in suffix_sources:
        return suffix_sources[suffix]

    with path.open("rb") as file:
        header = file.read(65536)

    if header.startswith(b"SQLite format 3\x00"):
        return "rsk"

    if header.startswith(b"PK\x03\x04"):
        return "datalakes"

    if (
        header.startswith(b"\x89HDF\r\n\x1a\n")
        or header.startswith((b"CDF\x01", b"CDF\x02", b"CDF\x05"))
    ):
        return "datalakes"

    stripped = header.lstrip()

    if stripped.startswith((b"{", b"[")):
        return "datalakes"

    decoded = []

    for encoding in ("utf-16", "utf-8", "latin-1"):
        try:
            decoded.append(header.decode(encoding))
        except UnicodeDecodeError:
            continue

    text = "\n".join(decoded)

    if (
        "KOR MEASUREMENT DATA FILE EXPORT" in text.upper()
        or (
            "TIME (HH:MM:SS)" in text
            and "DATE (MM/DD/YYYY)" in text
        )
    ):
        return "kor"

    if "; Datasets" in text:
        return "tob"

    raise ValueError(
        "Could not detect the data source. Set source to one of: "
        "datalakes, rsk, kor, tob"
    )


def read(path, source="auto", **kwargs):
    """Read a supported lake or profiler data file.

    This is the main PyLake entry point for data import. In most cases the user
    only needs to provide a file path; PyLake selects the appropriate reader.

    Method
    ----------
    Standard extensions are mapped directly to their readers. When an extension
    is unknown, the beginning of the file is inspected for SQLite, ZIP,
    NetCDF/HDF5, JSON, KOR, or Sea & Sun TOB signatures. ``source`` can be used
    to bypass automatic detection for ambiguous or incorrectly named files.

    Parameters
    ----------
    path : path_like
        Path to a DataLakes, RBR RSK, KOR, or Sea & Sun TOB file.
    source : {"auto", "datalakes", "rsk", "rbr", "kor", "tob"}, default: "auto"
        Data source to detect or force.
    **kwargs
        Additional arguments passed to the selected reader. For example,
        ``time_unit`` can be supplied for numeric DataLakes timestamps.

    Returns
    -------
    xarray.Dataset
        Dataset returned by the selected source-specific reader.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the source cannot be detected or is not supported.

    Examples
    ----------
    >>> data = read("profile.rsk")
    >>> data = read("profile.tob")
    >>> data = read("export.csv", source="kor")

    See Also
    ----------
    read_datalakes, read_rsk, read_kor, read_tob
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    normalized_source = str(source).lower().replace("-", "_").replace(" ", "_")

    aliases = {
        "rbr": "rsk",
        "sea_and_sun": "tob",
        "sea&sun": "tob",
    }

    normalized_source = aliases.get(
        normalized_source,
        normalized_source,
    )

    if normalized_source == "auto":
        normalized_source = _detect_source(path)

    readers = {
        "datalakes": read_datalakes,
        "rsk": read_rsk,
        "kor": read_kor,
        "tob": read_tob,
    }

    if normalized_source not in readers:
        raise ValueError(
            "source must be one of: auto, datalakes, rsk, rbr, kor, tob"
        )

    return readers[normalized_source](
        path,
        **kwargs,
    )


def datalakes_to_xarray(data, time_unit="s"):
    """Convert a DataLakes temperature payload to an xarray dataset.

    Method
    ----------
    DataLakes stores time, depth, and temperature under the keys ``x``, ``y``,
    and ``z``. Numeric timestamps are converted to UTC datetimes and then made
    timezone-naive for xarray compatibility. Temperature is oriented as
    ``(time, depth)``. Optional derived variables ``y1`` to ``y6`` are renamed
    to descriptive PyLake variable names.

    Parameters
    ----------
    data : dict or path_like
        Parsed DataLakes mapping or path to a JSON file. Required keys are
        ``x`` (time), ``y`` (depth), and ``z`` (temperature).
    time_unit : str, default: "s"
        Unit used to decode numeric timestamps, as accepted by
        :func:`pandas.to_datetime`.

    Returns
    -------
    xarray.Dataset
        Dataset containing ``temperature(time, depth)`` and any optional
        DataLakes diagnostic series ``y1`` through ``y6`` under descriptive
        variable names.

    Raises
    ------
    TypeError
        If ``data`` is neither a mapping nor a JSON path.
    ValueError
        If required keys are missing or temperature dimensions do not match
        the time and depth coordinates.

    Examples
    --------
    >>> payload = {"x": [0], "y": [1, 2], "z": [[12.0, 10.0]]}
    >>> datalakes_to_xarray(payload).temperature.shape
    (1, 2)
    """
    if isinstance(data, (str, Path)):
        with open(data, "r", encoding="utf-8") as f:
            data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError(
            "data must be a dictionary or a path to a JSON file"
        )

    if not all(k in data for k in ["x", "y", "z"]):
        raise ValueError(
            "DataLakes data must contain 'x', 'y', and 'z' keys"
        )

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
        raise ValueError(
            "DataLakes temperature data 'z' must be two-dimensional"
        )

    if Temp.shape == (len(depth), len(time)):
        Temp = Temp.T
    elif Temp.shape != (len(time), len(depth)):
        raise ValueError(
            "DataLakes temperature dimensions do not match time and depth"
        )

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
    """Read DataLakes JSON, NetCDF, or ZIP data.

    ZIP archives may contain multiple JSON or NetCDF files; their datasets are
    concatenated along time. JSON files are preferred when both formats occur.

    Method
    ----------
    JSON input is passed to :func:`datalakes_to_xarray`. NetCDF input is loaded
    with xarray and detached from the source file. ZIP archives are inspected
    for supported members, including members stored in subdirectories, and all
    resulting datasets are joined along their time dimension.

    Parameters
    ----------
    path : path_like
        Input ``.json``, ``.nc``, or ``.zip`` file.
    time_unit : str, default: "s"
        Unit for numeric JSON timestamps.

    Returns
    -------
    xarray.Dataset
        DataLakes variables with a ``source`` attribute equal to ``DataLakes``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the format is unsupported or a ZIP contains no supported files.
    """
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
        raise ValueError(
            "DataLakes files must use a .json, .nc, or .zip extension"
        )

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
            raise ValueError(
                "DataLakes ZIP archive contains no JSON or NetCDF files"
            )

    ds = xr.concat(
        datasets,
        dim="time"
    )

    ds.attrs["source"] = "DataLakes"

    return ds


def read_rsk(path):
    """Read an RBR ``.rsk`` SQLite database.

    Channel metadata are read from the ``channels`` table and observations
    from ``data``. Each channel becomes a time-indexed dataset variable with
    its original units and long name.

    Method
    ----------
    RSK files are SQLite databases. Channel definitions are read from the
    ``channels`` table and matched by position to columns in the ``data``
    table. Millisecond timestamps are converted to NumPy datetimes and channel
    metadata are attached as xarray attributes.

    Parameters
    ----------
    path : path_like
        Path to an RBR RSK database.

    Returns
    -------
    xarray.Dataset
        Sensor variables indexed by ``time`` and labelled with source ``RBR``.
    """
    path = Path(path)

    with sqlite3.connect(path) as connection:
        channels = pd.read_sql_query(
            "SELECT * FROM channels",
            connection
        )

        data = pd.read_sql_query(
            "SELECT * FROM data",
            connection
        )
    time = pd.to_datetime(
        data.iloc[:, 0],
        unit="ms"
    ).to_numpy()
    variables = {}
    for i, column in enumerate(data.columns[1:]):
        channel = channels.iloc[i]
        name = channel["shortName"]
        variables[name] = (
            "time",
            data[column].to_numpy()
        )

    ds = xr.Dataset(
        variables,
        coords={
            "time": time
        }
    )

    for i, name in enumerate(variables):
        ds[name].attrs["units"] = channels.iloc[i]["units"]
        ds[name].attrs["long_name"] = channels.iloc[i]["longName"]

    ds.attrs["source"] = "RBR"

    return ds

def read_kor(path):
    """Read a UTF-16 KOR profiler export.

    The first nine metadata rows are skipped, date and time columns are merged,
    and every numeric measurement column is retained.

    Method
    ----------
    The nine-line instrument header is skipped. Separate date and time columns
    are combined into one coordinate, while non-numeric metadata columns are
    omitted from the returned dataset.

    Parameters
    ----------
    path : path_like
        Path to the KOR text or CSV export.

    Returns
    -------
    xarray.Dataset
        Numeric sensor variables indexed by ``time`` and labelled ``KOR``.
    """
    path = Path(path)

    data = pd.read_csv(
        path,
        encoding="utf-16",
        skiprows=9
    )

    time = pd.to_datetime(
        data["DATE (MM/DD/YYYY)"].astype(str)
        + " "
        + data["TIME (HH:MM:SS)"].astype(str)
    ).to_numpy()

    variables = {}

    for column in data.columns:
        if column in [
            "TIME (HH:MM:SS)",
            "DATE (MM/DD/YYYY)"
        ]:
            continue

        if pd.api.types.is_numeric_dtype(data[column]):
            variables[column] = (
                "time",
                data[column].to_numpy()
            )

    ds = xr.Dataset(
        variables,
        coords={
            "time": time
        }
    )

    ds.attrs["source"] = "KOR"

    return ds

def read_tob(path):
    """Read a Sea & Sun CTD ``.tob`` text export.

    Method
    ----------
    The reader locates the ``; Datasets`` marker, skips the following header
    lines, and parses the first eleven whitespace-separated fields of every
    measurement record. Invalid numeric values become missing values. Date and
    time fields are combined using day-first parsing.

    Parameters
    ----------
    path : path_like
        Path to a Latin-1 encoded TOB file containing a ``; Datasets`` section.

    Returns
    -------
    xarray.Dataset
        Pressure, temperature, conductivity, chlorophyll, turbidity, pH, and
        oxygen observations indexed by time and labelled ``Sea & Sun``.

    Raises
    ------
    ValueError
        If the ``; Datasets`` section cannot be found.
    """
    path = Path(path)

    with open(
        path,
        "r",
        encoding="latin-1"
    ) as f:
        lines = f.readlines()

    start = None

    for i, line in enumerate(lines):
        if line.lstrip().startswith("; Datasets"):
            start = i + 3
            break

    if start is None:
        raise ValueError(
            "TOB file does not contain a '; Datasets' section"
        )

    rows = []

    for line in lines[start:]:
        values = line.split()

        if len(values) >= 11:
            rows.append(values[:11])

    data = pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "pressure",
            "temperature",
            "conductivity",
            "chlorophyll",
            "turbidity",
            "pH",
            "oxygen_saturation",
            "oxygen_mg_l",
            "date",
            "time"
        ]
    )

    numeric = [
        "dataset",
        "pressure",
        "temperature",
        "conductivity",
        "chlorophyll",
        "turbidity",
        "pH",
        "oxygen_saturation",
        "oxygen_mg_l"
    ]

    for column in numeric:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    time = pd.to_datetime(
        data["date"]
        + " "
        + data["time"],
        dayfirst=True
    ).to_numpy()

    ds = xr.Dataset(
        {
            "pressure": (
                "time",
                data["pressure"].to_numpy()
            ),
            "temperature": (
                "time",
                data["temperature"].to_numpy()
            ),
            "conductivity": (
                "time",
                data["conductivity"].to_numpy()
            ),
            "chlorophyll": (
                "time",
                data["chlorophyll"].to_numpy()
            ),
            "turbidity": (
                "time",
                data["turbidity"].to_numpy()
            ),
            "pH": (
                "time",
                data["pH"].to_numpy()
            ),
            "oxygen_saturation": (
                "time",
                data["oxygen_saturation"].to_numpy()
            ),
            "oxygen_mg_l": (
                "time",
                data["oxygen_mg_l"].to_numpy()
            )
        },
        coords={
            "time": time
        }
    )

    ds.attrs["source"] = "Sea & Sun"

    return ds
