"""Build the beginner-facing PyLake learning notebooks."""

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
    for index, cell in enumerate(cells, start=1):
        cell["id"] = f"cell-{index:03d}"
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


READING_CELLS = [
    markdown("""# 1 — Read and understand CTD files with PyLake

## Learning goals

At the end of this notebook, you will be able to:

1. explain what a CTD profile contains;
2. open four supported file formats with `pylake.read`;
3. understand the dimensions, coordinates, and variables in the result;
4. display measurements as a table and as a depth profile;
5. diagnose the most common reading errors.

No private dataset or internet connection is needed. The repository contains small reproducible files that you can safely modify."""),
    markdown("""## What is a CTD profile?

**CTD** means **Conductivity, Temperature, Depth**. A CTD instrument is lowered through the water and records measurements at successive depths. Depending on the instrument, a file may also contain pressure, oxygen, chlorophyll, turbidity, or pH.

In this tutorial, depth increases downward. A temperature profile therefore answers a simple question: **how does water temperature change from the surface to the bottom?**"""),
    markdown("""## Before running the notebook

Open the notebook from the PyLake repository and select the Python environment in which PyLake is installed. It works whether Jupyter starts in the repository root or in `notebooks/`. Execute the cells from top to bottom. If a cell fails, read the explanation immediately above it before changing the code."""),
    code("""from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import pylake

repository_root = Path.cwd()
if not (repository_root / "examples").is_dir():
    repository_root = repository_root.parent
if not (repository_root / "examples").is_dir():
    raise FileNotFoundError("Run this notebook from the PyLake repository or its notebooks directory.")
sys.path.insert(0, str(repository_root))

from examples.generate_ctd_examples import main as generate_examples

data_directory = generate_examples()
data_directory"""),
    markdown("""## The four example formats

PyLake supports several exporters because CTD manufacturers and data platforms do not all store measurements in the same way.

| Example | Origin | Main idea |
|---|---|---|
| `example_datalakes.json` | DataLakes | several temperature profiles on a time × depth grid |
| `example.rsk` | RBR | instrument database containing channels and metadata |
| `example_kor.csv` | KOR | text table with date, time, depth, and temperature |
| `example.tob` | Sea & Sun | text export containing several water-quality variables |

All four files contain deterministic synthetic measurements. They exist to teach and test the readers; they are not observations from a real lake."""),
    code("""example_files = sorted(data_directory.iterdir())
pd.DataFrame({
    "file": [path.name for path in example_files],
    "size_bytes": [path.stat().st_size for path in example_files],
})"""),
    markdown("""## Look at the file before using PyLake

A TOB file is ordinary text. Header lines begin with `;`, followed by rows of measurements. Looking at a few lines makes the reader less mysterious: PyLake is translating this instrument-specific layout into named scientific variables.

> **Good habit:** inspect a small part of an unfamiliar file, but never edit the original field file directly."""),
    code("""tob_preview = (data_directory / "example.tob").read_text(encoding="latin-1").splitlines()
print("\\n".join(tob_preview[:7]))"""),
    markdown("""## Start with one file

`pylake.read(path)` is the normal entry point. It examines the file and chooses the appropriate reader automatically."""),
    code("""tob_path = data_directory / "example.tob"
tob = pylake.read(tob_path)
tob"""),
    markdown("""## How to read an `xarray.Dataset`

The object displayed above is an `xarray.Dataset`:

- **Dimensions** give the size of each axis. Here, `time: 20` means 20 measurements.
- **Coordinates** identify positions on those axes, such as timestamps.
- **Data variables** contain the measurements, such as temperature and pressure.
- **Attributes** contain metadata describing the source.

Unlike an anonymous numerical matrix, a Dataset keeps names, coordinates, and metadata next to the values."""),
    code("""summary = {
    "dimensions": dict(tob.sizes),
    "coordinates": list(tob.coords),
    "variables": list(tob.data_vars),
    "detected_source": tob.attrs.get("source"),
}
summary"""),
    markdown("""## Turn selected variables into a readable table

The first five rows let us check whether the values and units look plausible before doing any scientific calculation."""),
    code("""tob_table = tob[["pressure", "temperature", "oxygen_mg_l"]].to_dataframe()
tob_table.head()"""),
    markdown("""## Plot the temperature profile

Scientific profiles conventionally place temperature on the horizontal axis and depth or pressure on the vertical axis. The vertical axis is inverted so the surface appears at the top."""),
    code("""fig, ax = plt.subplots(figsize=(5, 6))
ax.plot(tob["temperature"], tob["pressure"], marker="o")
ax.invert_yaxis()
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Pressure (dbar)")
ax.set_title("Synthetic Sea & Sun temperature profile")
ax.grid(alpha=0.3)
plt.show()"""),
    markdown("""### Interpretation

The example is warm near the surface and colder at depth. The strongest temperature change occurs around the middle of the profile. Later notebooks will estimate the thermocline associated with this transition."""),
    markdown("""## Read every format with the same public function

The purpose of `pylake.read` is to hide format-specific parsing from the user. The following loop opens every example and reports the detected source and available variables."""),
    code("""rows = []
datasets = {}

for path in example_files:
    dataset = pylake.read(path)
    datasets[path.name] = dataset
    rows.append({
        "file": path.name,
        "source": dataset.attrs.get("source"),
        "dimensions": dict(dataset.sizes),
        "variables": ", ".join(dataset.data_vars),
    })

pd.DataFrame(rows)"""),
    markdown("""## When should specialized readers be used?

Most users should call `pylake.read`. Specialized readers are useful when the format is already known or when automatic detection is being diagnosed.

| Function | Format |
|---|---|
| `read_datalakes` | DataLakes JSON, NetCDF, or ZIP |
| `read_rsk` | RBR `.rsk` |
| `read_kor` | KOR CSV export |
| `read_tob` | Sea & Sun TOB export |
| `datalakes_to_xarray` | DataLakes dictionary already loaded in memory |"""),
    code("""specialized = {
    "DataLakes": pylake.read_datalakes(data_directory / "example_datalakes.json"),
    "RBR": pylake.read_rsk(data_directory / "example.rsk"),
    "KOR": pylake.read_kor(data_directory / "example_kor.csv"),
    "Sea & Sun": pylake.read_tob(data_directory / "example.tob"),
}

{name: dict(dataset.sizes) for name, dataset in specialized.items()}"""),
    markdown("""## DataLakes dictionaries

A DataLakes export uses `x` for time, `y` for depth, and `z` for the temperature matrix. `datalakes_to_xarray` is useful when that dictionary has already been obtained from an API or decoded from JSON."""),
    code("""import json

datalakes_path = data_directory / "example_datalakes.json"
payload = json.loads(datalakes_path.read_text())
datalakes_from_memory = pylake.datalakes_to_xarray(payload)
datalakes_from_memory["temperature"].isel(time=0).to_dataframe().head()"""),
    markdown("""## Use your own file

Replace the path below with a real CTD file. Keeping the existence check makes the cell safe before a path has been selected."""),
    code("""my_file = Path("path/to/my/profile.rsk")

if my_file.exists():
    my_dataset = pylake.read(my_file)
    display(my_dataset)
else:
    print("Choose a real file path when you are ready.")"""),
    markdown("""## Understand common failures

- `FileNotFoundError`: Python cannot find the path.
- Unsupported-source `ValueError`: the format was not recognized.
- Shape error in DataLakes: the temperature matrix does not match time and depth.
- Parsing error: the real exporter uses a layout not represented by the reader.

When a real file fails, preserve the smallest shareable example that still reproduces the failure."""),
    code("""try:
    pylake.read("missing-profile.rsk")
except FileNotFoundError as error:
    print(f"Expected example: {type(error).__name__}: {error}")"""),
    markdown("""## Check your understanding

1. Change `example.tob` to `example.rsk` and identify the temperature and pressure variable names.
2. Display the last five rows rather than the first five.
3. Change the plot title and marker.
4. Explain, in one sentence, why `pylake.read` is preferable for a new user.

<details>
<summary><strong>Suggested answers</strong></summary>

1. The RSK temperature and pressure variables are `temp14` and `pres24`.
2. Replace `.head()` with `.tail()`.
3. Edit `ax.set_title(...)` and the `marker=` argument.
4. `pylake.read` detects the source automatically, so the user does not need to select a format-specific parser.

</details>

## Take-away

`pylake.read` provides one public interface for different CTD formats. Always inspect the Dataset, a few table rows, and a profile plot before starting scientific analysis."""),
]


PROFILE_CELLS = [
    markdown("""# 2 — Clean and analyse a lake temperature profile

## Learning goals

At the end of this notebook, you will be able to:

1. recognize common problems in a raw cast;
2. retain the downcast and average repeated depths;
3. estimate the thermocline and seasonal thermocline;
4. interpret the centre of buoyancy;
5. calculate volume-weighted temperatures for lake layers;
6. recognize important edge cases.

The notebook starts with small visible arrays so every transformation can be understood before applying it to a file."""),
    code("""import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylake

from pylake import functions as fn"""),
    markdown("""## A simple stratified profile

The upper water is warm, the deep water is cold, and the transition is rapid around 3–5 m. This is the typical structure of a stratified lake."""),
    code("""depth = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10], dtype=float)
temperature = np.array([21.0, 20.8, 20.3, 18.8, 14.0, 10.8, 9.6, 8.9, 8.6])

profile = pd.DataFrame({"depth_m": depth, "temperature_c": temperature})
profile"""),
    code("""fig, ax = plt.subplots(figsize=(5, 6))
ax.plot(temperature, depth, marker="o")
ax.invert_yaxis()
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Depth (m)")
ax.set_title("Stratified temperature profile")
ax.grid(alpha=0.3)
plt.show()"""),
    markdown("""## `depth_filter`: retain the downcast

A real instrument may remain near the surface, move upward briefly, and record an upcast after reaching maximum depth. `depth_filter` keeps the coherent descent. With `index=True`, it returns row positions that can be applied to every other sensor column."""),
    code("""raw_depth = np.array([0, 0.1, 0.05, 0.2, 0.8, 1.5, 1.4, 2.0, 3.0, 2.5])
raw_temperature = np.array([21.2, 21.1, 21.1, 21.0, 20.7, 19.8, 19.9, 18.0, 14.0, 15.2])

kept_rows = fn.depth_filter(raw_depth, run_length=2, index=True)
filtered = pd.DataFrame({
    "depth_m": raw_depth[kept_rows],
    "temperature_c": raw_temperature[kept_rows],
})

pd.DataFrame({"raw_depth_m": raw_depth, "raw_temperature_c": raw_temperature}), filtered"""),
    code("""fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=True)

axes[0].plot(raw_temperature, raw_depth, marker="o", color="0.45")
axes[0].set_title("Before: instrument movement")
axes[0].set_xlabel("Temperature (°C)")
axes[0].set_ylabel("Depth (m)")

axes[1].plot(filtered["temperature_c"], filtered["depth_m"], marker="o", color="tab:blue")
axes[1].set_title("After: retained downcast")
axes[1].set_xlabel("Temperature (°C)")

for axis in axes:
    axis.invert_yaxis()
    axis.grid(alpha=0.3)

fig.suptitle("Effect of depth_filter")
fig.tight_layout()
plt.show()"""),
    markdown("""### Interpretation

Returning indices is important: filtering only the depth vector would separate temperature, oxygen, and other measurements from their original rows."""),
    markdown("""## `depth_average`: merge repeated depths

Sensors may record several values at the same depth. `depth_average` returns unique depths and the mean finite measurement at each depth."""),
    code("""repeated_depth = np.array([1, 1, 2, 2, 3, 4, 4], dtype=float)
repeated_temperature = np.array([20, 22, 18, 20, 15, 11, 13], dtype=float)

average_depth, average_temperature = fn.depth_average(
    repeated_depth,
    repeated_temperature,
)

pd.DataFrame({"depth_m": average_depth, "mean_temperature_c": average_temperature})"""),
    markdown("""## Why density is used

The thermocline is estimated from the vertical change in **water density**, not directly from the largest temperature difference. Water density depends nonlinearly on temperature and can also depend on salinity."""),
    code("""density = np.asarray(pylake.water_density(temperature, S=0.2))

pd.DataFrame({
    "depth_m": depth,
    "temperature_c": temperature,
    "density_kg_m3": density,
}).head()"""),
    markdown("""## `thermocline`: locate the strongest density transition

The function returns two values: the estimated depth and the index of the density-gradient interval. `weighted=True` refines the location using neighbouring gradients. `weighted=False` returns the midpoint of the strongest interval."""),
    code("""weighted_depth, weighted_index = pylake.thermocline(
    temperature,
    depth,
    weighted=True,
)
midpoint_depth, midpoint_index = pylake.thermocline(
    temperature,
    depth,
    weighted=False,
)

pd.DataFrame({
    "method": ["weighted", "interval midpoint"],
    "thermocline_depth_m": [float(weighted_depth), float(midpoint_depth)],
    "gradient_interval_index": [int(weighted_index), int(midpoint_index)],
})"""),
    code("""fig, ax = plt.subplots(figsize=(5, 6))
ax.plot(temperature, depth, marker="o", label="temperature")
ax.axhline(float(weighted_depth), color="crimson", linestyle="--", label="thermocline")
ax.invert_yaxis()
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Depth (m)")
ax.set_title("Thermocline on the temperature profile")
ax.legend()
ax.grid(alpha=0.3)
plt.show()"""),
    markdown("""### Interpretation

The dashed line crosses the rapid transition between warm surface water and cold deep water. The returned depth is an estimate between sensors, not a claim that a sensor existed at exactly that depth."""),
    markdown("""## `seasonal_thermocline`: choose among several gradients

Some profiles contain a shallow daily gradient and a deeper seasonal gradient. The seasonal function searches qualifying density-gradient peaks and can select the deeper structure. Smoothing is disabled here because this is one short demonstration profile."""),
    code("""seasonal_depth, seasonal_index = pylake.seasonal_thermocline(
    temperature,
    depth,
    seasonal_smoothed=False,
    weighted=True,
)

float(seasonal_depth), int(seasonal_index)"""),
    markdown("""## `center_buoyancy`: centre of stratification

This quantity is the depth-weighted centre of positive buoyancy frequency. It summarizes where stable stratification is concentrated. A completely uniform profile has no meaningful centre and returns `NaN`."""),
    code("""stratified_center = fn.center_buoyancy(temperature, depth)
uniform_center = fn.center_buoyancy(np.full(depth.size, 10.0), depth)

{"stratified_profile_m": stratified_center, "uniform_profile": uniform_center}"""),
    markdown("""## Bathymetry and volume-weighted layer means

A simple arithmetic mean treats a narrow deep layer and a wide surface layer equally. PyLake can weight measurements using lake area at successive depths.

- `bthD` contains bathymetric depths.
- `bthA` contains lake area at those depths.
- Area normally decreases toward the bottom."""),
    code("""bthD = np.array([0, 2, 4, 6, 8, 10], dtype=float)
bthA = np.array([100, 92, 78, 55, 25, 0], dtype=float)

pd.DataFrame({"bathymetry_depth_m": bthD, "lake_area_relative": bthA})"""),
    markdown("""## Layer functions

`layer_temperature` calculates a volume-weighted mean between two boundaries. The convenience wrappers calculate the entire lake, the epilimnion above the metalimnion, and the hypolimnion below it."""),
    code("""meta_top = 3.0
meta_bottom = 5.0

layer_results = {
    "upper_0_to_3_m": fn.layer_temperature(0, meta_top, temperature, depth, bthA, bthD),
    "whole_lake": fn.whole_lake_temperature(temperature, depth, bthA, bthD),
    "epilimnion": fn.epi_temperature(temperature, depth, bthA, bthD, meta_top),
    "hypolimnion": fn.hypo_temperature(temperature, depth, bthA, bthD, meta_bottom),
}

pd.Series(layer_results, name="temperature_c")"""),
    code("""fig, ax = plt.subplots(figsize=(5, 6))
ax.plot(temperature, depth, color="black", marker="o", zorder=3)
ax.axhspan(0, meta_top, color="#f4a261", alpha=0.35, label="epilimnion")
ax.axhspan(meta_top, meta_bottom, color="#e9c46a", alpha=0.35, label="metalimnion")
ax.axhspan(meta_bottom, depth.max(), color="#457b9d", alpha=0.35, label="hypolimnion")
ax.invert_yaxis()
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Depth (m)")
ax.set_title("Thermal layers used in the calculations")
ax.legend(loc="lower left")
plt.show()"""),
    markdown("""`layer_density` performs the same volume-weighted calculation after converting temperature and salinity into density."""),
    code("""fn.layer_density(
    0,
    meta_top,
    temperature,
    depth,
    bthA,
    bthD,
    sal=0.2,
)"""),
    markdown("""## `ustar`: wind-driven water friction velocity

`ustar` converts wind speed into a friction velocity in water. It requires wind speed in m/s, wind measurement height in metres, and mean epilimnion density in kg/m³."""),
    code("""wind = np.array([2, 5, 10], dtype=float)
friction_velocity = fn.ustar(wind, wind_height=10, average_epi_density=998)

pd.DataFrame({"wind_m_s": wind, "ustar_m_s": friction_velocity})"""),
    markdown("""## Important edge cases

- Fewer than three measurements cannot define a reliable gradient profile.
- Repeated depths should be merged before thermocline analysis.
- A nearly uniform temperature profile is classified as mixed.
- Irregular depth spacing is supported.
- Layer boundaries must satisfy `top <= bottom`.
- Units must remain consistent."""),
    code("""edge_cases = {
    "three_points": pylake.thermocline([20, 10, 9], [1, 2, 3]),
    "uniform": pylake.thermocline([10, 10, 10, 10], [0, 1, 2, 3]),
    "irregular": pylake.thermocline(
        [21, 20.8, 20.4, 18, 12, 10, 9],
        [0, 0.4, 1.3, 2.7, 4.8, 7.5, 10],
    ),
}
edge_cases"""),
    markdown("""## Check your understanding

1. Make the surface water 2 °C warmer and recalculate the thermocline.
2. Compare weighted and midpoint results on irregular depths.
3. Move `meta_top` from 3 m to 4 m and explain the epilimnion-temperature change.
4. Increase wind speed and describe what happens to `ustar`.

<details>
<summary><strong>What you should observe</strong></summary>

1. Changing surface temperature alters density gradients and can move the weighted estimate.
2. The midpoint stays at the centre of one sensor interval; weighting can place the estimate elsewhere inside it.
3. A deeper `meta_top` includes more cool water and will generally reduce the epilimnion mean.
4. `ustar` increases with wind speed because stronger wind transfers more momentum to the water.

</details>

## Take-away

Clean the cast first, inspect it visually, estimate stratification, and only then calculate layer properties. A numerical result should always be interpreted together with the profile and its units."""),
]


WORKFLOW_CELLS = [
    markdown("""# 3 — Complete workflow: from a CTD file to lake indicators

This notebook combines reading, quality control, visualization, and profile analysis. It is intentionally shorter than the two teaching notebooks: use it as a reusable template after understanding the individual steps.

## Workflow

1. create or select a CTD file;
2. read it with the public interface;
3. identify depth and temperature;
4. clean the cast;
5. inspect the profile;
6. estimate thermocline and centre of buoyancy;
7. report results together with assumptions."""),
    code("""from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylake

repository_root = Path.cwd()
if not (repository_root / "examples").is_dir():
    repository_root = repository_root.parent
if not (repository_root / "examples").is_dir():
    raise FileNotFoundError("Run this notebook from the PyLake repository or its notebooks directory.")
sys.path.insert(0, str(repository_root))

from examples.generate_ctd_examples import main as generate_examples
from pylake import functions as fn"""),
    markdown("""## Step 1 — Select a reproducible file

We use the Sea & Sun example because it contains recognizable pressure and temperature variables. Replace the path later with a real `.tob`, `.rsk`, KOR CSV, or DataLakes export."""),
    code("""data_directory = generate_examples()
path = data_directory / "example.tob"
path"""),
    markdown("""## Step 2 — Read and inspect

Never assume variable names before inspecting the Dataset. Different formats use different channel names."""),
    code("""dataset = pylake.read(path)
dataset"""),
    code("""pd.DataFrame({
    "variable": list(dataset.data_vars),
    "dimensions": [str(dataset[name].dims) for name in dataset.data_vars],
    "units": [dataset[name].attrs.get("units", "not provided") for name in dataset.data_vars],
})"""),
    markdown("""## Step 3 — Extract the scientific variables

For this example, pressure is used as a depth proxy. For precise field analysis, convert pressure to depth when required by the instrument protocol and document the conversion."""),
    code("""raw_depth = np.asarray(dataset["pressure"], dtype=float)
raw_temperature = np.asarray(dataset["temperature"], dtype=float)

raw = pd.DataFrame({
    "depth_or_pressure": raw_depth,
    "temperature_c": raw_temperature,
})
raw.head()"""),
    markdown("""## Step 4 — Quality control

The downcast indices are applied to both variables. Repeated depths are then averaged. Keeping these two operations explicit makes the cleaning decisions auditable."""),
    code("""kept_rows = fn.depth_filter(raw_depth, run_length=3, index=True)
downcast_depth = raw_depth[kept_rows]
downcast_temperature = raw_temperature[kept_rows]

clean_depth, clean_temperature = fn.depth_average(
    downcast_depth,
    downcast_temperature,
)

clean = pd.DataFrame({
    "depth_or_pressure": clean_depth,
    "temperature_c": clean_temperature,
})
clean.head()"""),
    code("""quality_control = pd.Series({
    "source": dataset.attrs.get("source"),
    "raw_measurements": raw_depth.size,
    "retained_downcast_measurements": downcast_depth.size,
    "unique_clean_depths": clean_depth.size,
    "missing_temperatures": int(np.isnan(raw_temperature).sum()),
})
quality_control"""),
    markdown("""## Step 5 — Plot before calculating

A plot can reveal inverted axes, impossible values, gaps, spikes, or a profile that is actually mixed."""),
    code("""fig, ax = plt.subplots(figsize=(5, 6))
ax.plot(clean_temperature, clean_depth, marker="o")
ax.invert_yaxis()
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Pressure / depth proxy")
ax.set_title(path.name)
ax.grid(alpha=0.3)
plt.show()"""),
    markdown("""## Step 6 — Calculate indicators

The thermocline identifies the strongest density transition. The centre of buoyancy summarizes where stratification is concentrated."""),
    code("""thermocline_depth, thermocline_index = pylake.thermocline(
    clean_temperature,
    clean_depth,
    weighted=True,
)
buoyancy_center = fn.center_buoyancy(clean_temperature, clean_depth)

results = pd.Series({
    "thermocline_depth": float(thermocline_depth),
    "thermocline_interval_index": int(thermocline_index),
    "center_buoyancy_depth": float(buoyancy_center),
})
results"""),
    markdown("""## A compact result table

The final output should be easy to inspect, export, and compare across profiles. A result should travel together with basic quality-control counts rather than as an isolated number."""),
    code("""report = pd.DataFrame([{
    "file": path.name,
    "source": dataset.attrs.get("source"),
    "raw_measurements": raw_depth.size,
    "clean_measurements": clean_depth.size,
    "thermocline_depth": float(thermocline_depth),
    "center_buoyancy_depth": float(buoyancy_center),
    "weighted_thermocline": True,
}])
report"""),
    code("""fig, ax = plt.subplots(figsize=(5, 6))
ax.plot(clean_temperature, clean_depth, marker="o", label="clean profile")
ax.axhline(float(thermocline_depth), color="crimson", linestyle="--", label="thermocline")
ax.axhline(float(buoyancy_center), color="navy", linestyle=":", label="centre of buoyancy")
ax.invert_yaxis()
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Pressure / depth proxy")
ax.set_title("Interpreted CTD profile")
ax.legend()
ax.grid(alpha=0.3)
plt.show()"""),
    markdown("""## Step 7 — Interpret and report

For this synthetic profile, warm surface water overlies colder deep water. The thermocline marks the strongest transition. The centre of buoyancy describes the overall centre of stable stratification and therefore need not equal the thermocline.

A real report should state:

- file and instrument source;
- date and location;
- depth or pressure units;
- temperature units;
- cleaning parameters;
- salinity assumption;
- whether weighted thermocline refinement was used;
- any missing values or rejected measurements."""),
    markdown("""## Reusable function

The following small function packages the demonstrated workflow. It deliberately requires explicit variable names because real instruments do not all use the same channel labels."""),
    code("""def analyse_profile(path, depth_variable, temperature_variable, run_length=3):
    dataset = pylake.read(path)
    raw_depth = np.asarray(dataset[depth_variable], dtype=float)
    raw_temperature = np.asarray(dataset[temperature_variable], dtype=float)

    kept = fn.depth_filter(raw_depth, run_length=run_length, index=True)
    depth, temperature = fn.depth_average(raw_depth[kept], raw_temperature[kept])
    thermo_depth, thermo_index = pylake.thermocline(temperature, depth)

    return {
        "dataset": dataset,
        "depth": depth,
        "temperature": temperature,
        "thermocline_depth": float(thermo_depth),
        "thermocline_index": int(thermo_index),
        "center_buoyancy": float(fn.center_buoyancy(temperature, depth)),
    }

analysis = analyse_profile(path, "pressure", "temperature")
{key: value for key, value in analysis.items() if key not in {"dataset", "depth", "temperature"}}"""),
    markdown("""## Try another format

For the RBR example, the corresponding variables are `pres24` and `temp14`."""),
    code("""rsk_analysis = analyse_profile(
    data_directory / "example.rsk",
    depth_variable="pres24",
    temperature_variable="temp14",
)

{key: value for key, value in rsk_analysis.items() if key not in {"dataset", "depth", "temperature"}}"""),
    markdown("""## Final checklist

Before trusting a result, confirm:

- [ ] the correct file and source were detected;
- [ ] depth increases in the intended direction;
- [ ] units are known;
- [ ] temperature values are plausible;
- [ ] the retained downcast was inspected;
- [ ] repeated depths were handled;
- [ ] the thermocline is visible on the plot;
- [ ] assumptions are recorded.

The code produces numbers; the plot, metadata, and documented assumptions make those numbers scientifically interpretable."""),
]


def main():
    output = Path(__file__).resolve().parents[1] / "notebooks"
    output.mkdir(exist_ok=True)
    paths = {
        output / "IO_demo.ipynb": notebook(READING_CELLS),
        output / "profiles_demo.ipynb": notebook(PROFILE_CELLS),
        output / "workflow_demo.ipynb": notebook(WORKFLOW_CELLS),
    }
    for path, content in paths.items():
        path.write_text(json.dumps(content, indent=1), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
