import numpy as np
import pandas as pd

data = np.random.normal(loc=27000, scale=15000, size=100)
incomes = pd.Series(data)

binned_incomes, bins = pd.qcut(
    incomes,
    q=4,
    labels=["Low", "Medium-Low", "Medium-High", "High"],
    retbins=True,
)

print("--- 1. Summary Statistics of Original Data ---")
print(incomes.describe())
print("\n" + "=" * 40 + "\n")

print("--- 2. Quantile Binning Result (Value Counts) ---")
print(binned_incomes.value_counts().sort_index())
print("\n" + "=" * 40 + "\n")

print("--- 3. Quantile Boundaries (Bins) ---")
print(f"Boundaries (Bins): {bins}")
