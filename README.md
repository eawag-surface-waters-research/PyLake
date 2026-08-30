# PyLake

This work present methods used to compute meaningful physical properties in aquatic sciences.

The methods are based on Xarray. 
Multi-dimensional large time-series array are compatible if an xarray is passed as input.

Algorithms and documentation are sometimes inspired by LakeAnalyzer in R (https://github.com/GLEON/rLakeAnalyzer)

Implemented methods:
* Thermocline
* Mixed layer
* Metalimnion extent (epilimnion and hypolimnion depth)
* Wedderburn Number
* Schmidt stability
* Heat content
* Seiche periode
* Lake Number
* Brunt-Vaisala frequency
* Average layer temperature
* Monin-Obhukov 

## Installation

Pylake use Dask which require a python version >=3.8

`pip install pylake`

## Usage


Have a look in the notebooks, an example is provided

```python
import pylake
import numpy as np

Temp = np.array([14.3,14,12.1,10,9.7,9.5,6,5])
depth = np.array([1,2,3,4,5,6,7,8])
epilimnion, hypolimnion = pylake.metalimnion(temp, depth)
```

## CTD data workflow

PyLake can turn common CTD exports into analysis-ready `xarray.Dataset`
objects through a single interface:

```python
import pylake

dataset = pylake.read("examples/data/example.tob")
print(dataset)
```

The workflow supports DataLakes, RBR RSK, KOR, and Sea & Sun TOB files, then
connects file reading with profile cleaning, thermocline detection, layer
statistics, and lake-metabolism calculations.

Start with one of the guided notebooks:

- [`IO_demo.ipynb`](notebooks/IO_demo.ipynb) — read and understand CTD files;
- [`profiles_demo.ipynb`](notebooks/profiles_demo.ipynb) — clean and analyse a temperature profile;
- [`workflow_demo.ipynb`](notebooks/workflow_demo.ipynb) — run the complete workflow.

See [CTD analysis workflow](doc/CTD_WORKFLOW.md) for supported formats,
examples, validation, and reproducibility details.

## Work in progress

Warning messages and the lake-metabolism API remain under active development.
