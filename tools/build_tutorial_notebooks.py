"""Build the beginner-facing PyLake tutorial notebooks."""

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


IO_CELLS = [
    markdown("""# Reading CTD files with PyLake

This notebook is for someone who has never used this project. It explains the public reader, each specialized reader, the expected output, and what to do when a real file is unavailable. Run it from the root of the PyLake repository."""),
    markdown("""## 1. Imports and reproducible example files

The generator creates four tiny deterministic CTD files with 20 measurements. This keeps the tutorial usable without a private dataset or an internet connection."""),
    code("""from pathlib import Path
import pylake
from examples.generate_ctd_examples import main as generate_examples

data_directory = generate_examples()
sorted(path.name for path in data_directory.iterdir())"""),
    markdown("""## 2. `pylake.read`

`read(path)` is the recommended entry point. It detects DataLakes JSON/NetCDF/ZIP, RBR RSK, KOR CSV, and Sea & Sun TOB files. It returns an `xarray.Dataset`; dimensions describe the axes, variables contain observations, and `attrs['source']` records the detected source."""),
    code("""datasets = {}
for path in sorted(data_directory.iterdir()):
    datasets[path.name] = pylake.read(path)
    print(path.name, datasets[path.name].attrs["source"], dict(datasets[path.name].sizes))"""),
    markdown("""## 3. `read_datalakes` and `datalakes_to_xarray`

DataLakes stores time in `x`, depth in `y`, and the temperature matrix in `z`. `datalakes_to_xarray` converts an in-memory dictionary; `read_datalakes` opens JSON, NetCDF, or ZIP exports."""),
    code("""import json

json_path = data_directory / "example_datalakes.json"
payload = json.loads(json_path.read_text())
from_dictionary = pylake.datalakes_to_xarray(payload)
from_file = pylake.read_datalakes(json_path)
from_file"""),
    markdown("""The temperature variable uses `(time, depth)`. Select one profile with `.isel(time=0)` and inspect its physical coordinates."""),
    code("""from_file.temperature.isel(time=0).to_dataframe().head()"""),
    markdown("""## 4. `read_rsk`

RBR `.rsk` files are SQLite databases. The reader maps channels to dataset variables and preserves units and long names."""),
    code("""rsk = pylake.read_rsk(data_directory / "example.rsk")
rsk[["temp14", "pres24"]]"""),
    markdown("""## 5. `read_kor`

KOR exports are UTF-16 CSV files with a nine-line header. Numeric columns become dataset variables and the date/time columns become one time coordinate."""),
    code("""kor = pylake.read_kor(data_directory / "example_kor.csv")
kor"""),
    markdown("""## 6. `read_tob`

Sea & Sun TOB files contain a `; Datasets` marker followed by pressure, temperature, conductivity, chlorophyll, turbidity, pH, and oxygen measurements."""),
    code("""tob = pylake.read_tob(data_directory / "example.tob")
tob[["pressure", "temperature", "oxygen_mg_l"]]"""),
    markdown("""## 7. Forcing the source

Automatic detection is normally enough. Use `source=` only when an extension is missing or misleading."""),
    code("""forced = pylake.read(json_path, source="datalakes")
forced.attrs["source"]"""),
    markdown("""## 8. Understand failures

- `FileNotFoundError`: the path is wrong.
- `ValueError` during detection: use a supported file or pass `source=`.
- Shape error in DataLakes: `z` must match time × depth or depth × time.
- Parsing error in KOR/TOB: verify the exporter header and encoding.

When a real file fails, keep the smallest shareable failing sample. If sharing is impossible, regenerate the 20-point examples above and describe exactly how the real file differs."""),
    code("""try:
    pylake.read("missing-profile.rsk")
except FileNotFoundError as error:
    print(type(error).__name__, error)"""),
]


FUNCTION_CELLS = [
    markdown("""# New profile-analysis functions in PyLake

Every function below has an explanation, a minimal executable demonstration, and an interpretation. Public analysis functions come first; internal helpers are shown later so contributors can understand the implementation."""),
    code("""import numpy as np
import xarray as xr
import pylake
from pylake import functions as fn

depth = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
temperature = np.array([14.3, 14, 12.1, 10, 9.7, 9.5, 6, 5], dtype=float)
bthD = np.array([0, 2.3, 2.5, 4.2, 5.8, 8], dtype=float)
bthA = np.array([100, 90, 86, 82, 20, 0], dtype=float)"""),
    markdown("""## 1. `depth_filter`

CTDs often record a soak near the surface, small upward movements, and a final upcast. `depth_filter` retains the monotonic downcast. With `index=True`, it returns original row indices so all other sensor columns can be filtered identically."""),
    code("""raw_depth = np.array([0, 0.1, 0.05, 0.2, 0.8, 1.5, 1.4, 2.0, 3.0, 2.5])
filtered_depth = fn.depth_filter(raw_depth, run_length=2)
kept_rows = fn.depth_filter(raw_depth, run_length=2, index=True)
filtered_depth, kept_rows"""),
    markdown("""## 2. `depth_average`

Repeated depths are collapsed and their finite measurements are averaged. This is useful before interpolation and stratification calculations."""),
    code("""fn.depth_average([1, 1, 2, 2, 3], [20, 22, 18, 20, 15])"""),
    markdown("""## 3. `thermocline` and `weighted`

The maximum density-gradient interval identifies the thermocline. `weighted=True` (default) refines the position between sensors using Read et al. (2011). `weighted=False` returns the interval midpoint and is useful for debugging and direct interval comparisons."""),
    code("""weighted_depth, weighted_index = pylake.thermocline(temperature, depth)
midpoint_depth, midpoint_index = pylake.thermocline(temperature, depth, weighted=False)
weighted_depth, weighted_index, midpoint_depth, midpoint_index"""),
    markdown("""## 4. `seasonal_thermocline`

This selects the deepest density-gradient peak above `Smin`, falling back to the diurnal thermocline when no peak qualifies. `seasonal_smoothed=False` avoids time-series smoothing for a single profile."""),
    code("""pylake.seasonal_thermocline(
    temperature,
    depth,
    seasonal_smoothed=False,
    weighted=True,
)"""),
    markdown("""## 5. `center_buoyancy`

This is the depth-weighted center of positive buoyancy frequency. A stratified profile returns a depth; a uniform profile returns `NaN`."""),
    code("""fn.center_buoyancy(temperature, depth), fn.center_buoyancy(np.full(8, 10.0), depth)"""),
    markdown("""## 6. Layer averages

`layer_average` interpolates both observations and lake area on a fine vertical grid and calculates a volume-weighted mean. `layer_temperature` applies it directly to temperature; `layer_density` first converts temperature and salinity to density."""),
    code("""layer_value = fn.layer_average(0, 4, temperature, depth, bthA, bthD)
layer_temp = fn.layer_temperature(0, 4, temperature, depth, bthA, bthD)
layer_rho = fn.layer_density(0, 4, temperature, depth, bthA, bthD, sal=0.2)
layer_value, layer_temp, layer_rho"""),
    markdown("""## 7. Whole-lake, epilimnion, and hypolimnion temperature

These wrappers make layer boundaries explicit. `meta_top` is the top of the metalimnion and `meta_bottom` is its bottom."""),
    code("""whole = fn.whole_lake_temperature(temperature, depth, bthA, bthD)
epi = fn.epi_temperature(temperature, depth, bthA, bthD, meta_top=2.5)
hypo = fn.hypo_temperature(temperature, depth, bthA, bthD, meta_bottom=5)
whole, epi, hypo"""),
    markdown("""## 8. `ustar`

`ustar` converts wind speed into water friction velocity. Inputs are wind speed in m/s, measurement height in m, and mean epilimnion density in kg/m³."""),
    code("""fn.ustar(wind_speed=[2, 5, 10], wind_height=10, average_epi_density=998)"""),
    markdown("""## 9. Input helpers: `control`, `format_Temp`, and `to_xarray`

These internal utilities validate a minimum of three unique depths, orient arrays as time × depth, and attach named coordinates."""),
    code("""formatted = fn.format_Temp(depth, temperature)
data_array, normalized_depth = fn.to_xarray(temperature, depth)
fn.control(data_array, normalized_depth), formatted.shape, data_array"""),
    markdown("""## 10. Smoothing helpers

`smooth_1D` handles one vector. `smooth_temp` operates along the named depth axis and therefore also supports several profiles."""),
    code("""noisy = temperature + np.array([0, 0.1, -0.2, 0.2, -0.1, 0.1, -0.2, 0])
smoothed_vector = fn.smooth_1D(noisy, {"window_size": 5, "order": 2})
smoothed_array = fn.smooth_temp(data_array, depth, {"window_size": 5, "order": 2})
smoothed_vector, smoothed_array"""),
    markdown("""## 11. `weighted_method` and `find_peak_index`

`weighted_method` refines a density-gradient interval. `find_peak_index` returns the deepest peak above a threshold or a supplied fallback. They are implementation helpers used by the thermocline functions."""),
    code("""density = pylake.water_density(data_array, 0.2)
gradient = density.diff("depth") / density.depth.diff("depth")
interval_index = gradient.argmax("depth")
refined = fn.weighted_method(depth, density, interval_index)
peak = fn.find_peak_index([0, 0.3, 0, 0.7, 0], 0.1, 0)
refined, peak"""),
    markdown("""## 12. Remaining array and bathymetry helpers

`find_nearest_index` and `find_nearest` locate sensors. `set_nan` transfers a missing-value mask. `round_up_to_odd` produces valid smoothing windows. `check_bathy` aligns the deepest temperature and bathymetry bounds."""),
    code("""nearest_index = fn.find_nearest_index(depth, 3.6)
nearest_depth = fn.find_nearest(depth, 3.6)
masked = fn.set_nan(np.array([1, np.nan, 3]), np.array([10.0, 20.0, 30.0]))
odd_window = fn.round_up_to_odd(6)
checked = fn.check_bathy(temperature.reshape(1, -1), bthA, bthD, depth)
nearest_index, nearest_depth, masked, odd_window, [np.shape(value) for value in checked]"""),
    markdown("""## 13. Important edge cases

- Fewer than three measurements: warning and `NaN`.
- Repeated depths: clean with `depth_average` first.
- Uniform profile: no significant thermocline when the temperature range is below `mixed_cutoff`.
- `top > bottom`: layer functions raise `ValueError`.
- Irregular spacing: supported; weighted and midpoint estimates can differ.
- Multiple profiles: pass a 2-D time × depth array and matching timestamps."""),
    code("""cases = {
    "three points": pylake.thermocline([20, 10, 9], [1, 2, 3]),
    "uniform": pylake.thermocline([10, 10, 10, 10], [0, 1, 2, 3]),
    "irregular": pylake.thermocline([21, 20.8, 20.4, 18, 12, 10, 9], [0, 0.4, 1.3, 2.7, 4.8, 7.5, 10]),
}
cases"""),
]


def main():
    output = Path(__file__).resolve().parents[1] / "notebooks"
    output.mkdir(exist_ok=True)
    paths = {
        output / "IO_readers_tutorial.ipynb": notebook(IO_CELLS),
        output / "profile_functions_tutorial.ipynb": notebook(FUNCTION_CELLS),
    }
    for path, content in paths.items():
        path.write_text(json.dumps(content, indent=1), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
