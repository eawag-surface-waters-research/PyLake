# PyLake improvement roadmap

## High priority

1. Define a stable public API and export only documented functions.
2. Add schema validation and informative errors to every CTD reader.
3. Maintain small, versioned fixtures for every supported data source.
4. Run Python tests across supported Python, NumPy, and xarray versions in CI.
5. Turn the rLakeAnalyzer comparison into a regression job with stored reference
   outputs and explicit numeric tolerances.

## Numerical reliability

1. Document density, salinity, interpolation, and boundary conventions.
2. Add property-based tests for monotonic and irregular profiles.
3. Define behavior for missing data, repeated depths, inverted casts, ties, and
   profiles containing fewer than three valid measurements.
4. Compare thermocline, seasonal thermocline, layer temperatures, center of
   buoyancy, Schmidt stability, Lake Number, and friction velocity with
   rLakeAnalyzer.
5. Publish a validation table showing agreement, tolerance, and the reason for
   every intentional difference.

## Data ingestion

1. Normalize variable names and units across RSK, KOR, TOB, and DataLakes.
2. Add provenance attributes for source file, reader, and parser version.
3. Support file-like objects in addition to paths.
4. Detect malformed headers early and report the exact missing field.
5. Add optional downcast filtering and depth averaging after import.

## Documentation and usability

1. Keep one beginner notebook for readers and one for physical calculations.
2. Give every public function parameters, units, output shape, example, and edge
   cases in its docstring.
3. Add a short workflow: read, clean, average, calculate, validate, and export.
4. Build the API documentation automatically and run doctests in CI.
5. Add contribution instructions for tests, numerical tolerances, and sample
   data privacy.
