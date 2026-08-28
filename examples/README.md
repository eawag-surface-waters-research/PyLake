# Reproducible CTD examples

Run the generator from the project root:

```bash
python examples/generate_ctd_examples.py
```

It creates four deterministic files with 20 measurements in `examples/data/`:

- `example_datalakes.json`
- `example_kor.csv`
- `example.tob`
- `example.rsk`

All four files can be opened through `pylake.read`. The generator is the
fallback when a real CTD file is unavailable or an external download changes.
