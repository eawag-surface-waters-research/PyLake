# PyLake vs rLakeAnalyzer

The comparison uses identical depth and temperature vectors on both sides.
`rLakeAnalyzer::thermo.depth(..., seasonal = FALSE)` is compared with
`pylake.thermocline(..., weighted=True)`.

```bash
Rscript validation/run_rlakeanalyzer.R
python validation/compare_thermocline.py
```

The Python step creates `validation/comparison.csv` and reports the absolute
difference for every profile. A difference should be investigated in this
order:

1. density equation and salinity default;
2. weighted interpolation between sensors;
3. handling of missing or repeated depths;
4. tie-breaking at profile boundaries;
5. mixed-profile cutoff.

R and `rLakeAnalyzer` are intentionally not Python package dependencies. The R
script stops with an installation instruction when the package is absent.
