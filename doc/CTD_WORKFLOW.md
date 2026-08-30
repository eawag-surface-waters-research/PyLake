# From a CTD file to lake indicators

PyLake provides a reproducible path from instrument output to interpretable
lake-physics results. This guide introduces the workflow, shows where each
function belongs, and points to executable examples.

> **Scientific question**
> Given a raw CTD cast, where is the strongest density transition, how are the
> lake layers organized, and which measurements were retained during cleaning?

## The workflow at a glance

```text
CTD file → read → inspect → clean → calculate → visualize → report
```

| Stage | PyLake interface | Result |
|---|---|---|
| Read | `pylake.read` | An `xarray.Dataset` with source metadata |
| Inspect | Dataset dimensions, coordinates, and variables | Identified depth and measurement channels |
| Clean | `depth_filter`, `depth_average` | A monotonic profile with repeated depths combined |
| Calculate | `thermocline`, `center_buoyancy`, layer functions | Physical lake indicators |
| Validate | tests and rLakeAnalyzer comparison | Reproducible numerical checks |
| Communicate | guided notebooks and plots | An inspectable scientific result |

## 1. Read the file

`pylake.read` is the recommended public entry point. It detects the source
when possible and dispatches to the appropriate specialized reader.

```python
import pylake

dataset = pylake.read("examples/data/example.tob")
print(dataset.attrs["source"])
print(dataset.data_vars)
```

### Supported sources

| Source | Typical input | Specialized reader |
|---|---|---|
| DataLakes | JSON, NetCDF, or ZIP | `read_datalakes` |
| RBR | SQLite-based `.rsk` | `read_rsk` |
| KOR | UTF-16 CSV export | `read_kor` |
| Sea & Sun | `.tob` text export | `read_tob` |

Use a specialized reader when the source is already known. Use
`pylake.read(path, source=...)` when a missing or misleading extension prevents
automatic detection.

## 2. Inspect before calculating

Instrument formats do not use identical variable names. Inspect the returned
dataset before choosing the depth and temperature channels.

```python
print(dataset.sizes)
print(dataset.coords)
print(dataset.data_vars)
```

A reliable analysis records at least:

- the detected source;
- the original number of measurements;
- the selected depth and temperature variables;
- the number of observations retained after cleaning;
- whether thermocline interval weighting was enabled.

## 3. Clean the cast

Raw profiles may include measurements recorded before descent, during ascent,
or repeatedly at the same depth.

```python
from pylake import functions as fn

clean_depth, retained = fn.depth_filter(depth, index=True)
clean_temperature = temperature[retained]
clean_depth, clean_temperature = fn.depth_average(
    clean_depth,
    clean_temperature,
)
```

Plot the profile after cleaning. Depth should normally increase down the axis,
and unexpected gaps or spikes should be investigated before interpretation.

## 4. Calculate physical indicators

```python
thermocline_depth, thermocline_index = pylake.thermocline(
    clean_temperature,
    clean_depth,
    weighted=True,
)

buoyancy_center = fn.center_buoyancy(
    clean_temperature,
    clean_depth,
)
```

The thermocline marks the strongest density transition. The centre of
buoyancy summarizes where stratification is concentrated. Layer functions can
then estimate temperature, density, and averages for the whole lake,
epilimnion, or hypolimnion when suitable bathymetry and layer bounds are
available.

### Irregularly spaced depths

`weighted=True` accounts for unequal vertical intervals during thermocline
estimation. This is important when sensors are not sampled on a regular depth
grid. The unweighted mode remains available for comparison and compatibility.

## 5. Continue to lake metabolism

PyLake also provides oxygen and metabolism calculations:

- `o2_at_sat` for oxygen saturation;
- `k_crusius` for gas-exchange velocity;
- `metab_bookkeep` for bookkeeping estimates;
- `metab_ols` for ordinary least-squares estimates;
- `metab_mle` for maximum-likelihood estimates.

These functions require correctly aligned time series and physically
consistent units. Inspect their docstrings before substituting field data.

## Learn by running the examples

The three notebooks are ordered as a learning path rather than as an API
catalogue.

| Notebook | Start here when... | Outcome |
|---|---|---|
| [`IO_demo.ipynb`](../notebooks/IO_demo.ipynb) | you have never opened a CTD file with PyLake | read four formats and understand the returned dataset |
| [`profiles_demo.ipynb`](../notebooks/profiles_demo.ipynb) | you already have depth and temperature arrays | clean, visualize, and interpret a profile |
| [`workflow_demo.ipynb`](../notebooks/workflow_demo.ipynb) | you want the complete analysis sequence | produce indicators with quality-control context |

Each notebook contains learning goals, a worked example, interpretation,
exercises, solutions, and saved outputs.

## Reproducible CTD examples

Small deterministic files are stored in `examples/data/`. They represent all
supported reader families and contain only 20 measurements, making failures
easy to inspect and share.

Regenerate them with:

```bash
python examples/generate_ctd_examples.py
```

The generated files are teaching and test fixtures, not observations from a
real lake. When a private field file fails, reduce it to the smallest
shareable example while preserving the header and the failing rows.

## Scientific validation

Thermocline results are compared with `rLakeAnalyzer` on reference, mixed,
irregular, three-point, and multiple-gradient profiles.

```bash
Rscript validation/run_rlakeanalyzer.R
python validation/compare_thermocline.py
```

The irregular-depth comparison differs by approximately `0.0023 m`; all
included profiles agree within the configured tolerance. This comparison is a
numerical cross-check, not a claim that every preprocessing decision must be
identical between the Python and R workflows.

## Verify the installation

Run the complete suite from the repository root:

```bash
python -m pytest test -q
```

The current workflow is covered by 75 tests across file readers, reproducible
examples, profile processing, thermocline edge cases, layer calculations, and
lake metabolism.

## Working with a real profile

Before accepting a result, check the following:

- the file source was detected correctly;
- the selected variables have the expected units;
- depth is ordered consistently;
- missing and duplicate measurements were handled deliberately;
- the cleaned profile was plotted;
- weighting matches the sampling geometry;
- calculated indicators are reported with their assumptions.

The goal is not only to obtain a number, but to keep the path from instrument
file to scientific interpretation visible and reproducible.
