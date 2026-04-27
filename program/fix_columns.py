import pandas as pd
import numpy as np

fieldnames = [
    "name",
    "N",
    "J",
    "noise",
    "periodic",
    "beta",
    "gamma",
    "mean_degree",  # must be inserted here
    "r_mean",
    "alpha",
    "Tc",
    "participation_fraction",
    "avg_internal_degree",
    "n_components",
    "largest_component_fraction",
]

rows = []

with open("../logs/log_sizes.csv", "r") as f:
    next(f)
    for line in f:
        parts = line.strip().split(",")

        if len(parts) == len(fieldnames) - 1:
            # Old row → missing mean_degree
            parts.insert(7, 20)

        elif len(parts) != len(fieldnames):
            print("⚠️ Unexpected row length:", len(parts))
            continue  # or handle differently

        rows.append(parts)

df = pd.DataFrame(rows, columns=fieldnames)

# Optional: convert numeric columns
df = df.apply(pd.to_numeric, errors="ignore")

df.to_csv("../logs/cleaned_file.csv", index=False)
