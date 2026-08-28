"""Compare PyLake thermocline depths with rLakeAnalyzer results."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import pylake


def python_results(profiles):
    rows = []
    for name, current in profiles.groupby("profile", sort=True):
        depth, _ = pylake.thermocline(
            current["temperature"].to_numpy(),
            current["depth"].to_numpy(),
            weighted=True,
        )
        rows.append({"profile": name, "python_thermocline": depth})
    return pd.DataFrame(rows)


def explanation(row, tolerance):
    if np.isnan(row["python_thermocline"]) and np.isnan(row["r_thermocline"]):
        return "both classify the profile as mixed"
    if row["absolute_difference"] <= tolerance:
        return "agreement within tolerance"
    return (
        "inspect density formula, interval weighting, missing-value handling, "
        "and boundary tie-breaking"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", default="validation/profiles.csv")
    parser.add_argument("--r-results", default="validation/rlakeanalyzer_results.csv")
    parser.add_argument("--output", default="validation/comparison.csv")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    profiles = pd.read_csv(args.profiles)
    r_results_path = Path(args.r_results)
    if not r_results_path.exists():
        raise FileNotFoundError(
            f"{r_results_path} does not exist. Run Rscript "
            "validation/run_rlakeanalyzer.R first."
        )

    comparison = python_results(profiles).merge(
        pd.read_csv(r_results_path),
        on="profile",
        validate="one_to_one",
    )
    comparison["absolute_difference"] = (
        comparison["python_thermocline"] - comparison["r_thermocline"]
    ).abs()
    comparison["interpretation"] = comparison.apply(
        explanation,
        axis=1,
        tolerance=args.tolerance,
    )
    comparison.to_csv(args.output, index=False)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
