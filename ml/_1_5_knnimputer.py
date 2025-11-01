import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

data = {
    "customer_id": [1, 2, 3, 4, 5],
    "age": [35, 28, 45, 32, 50],
    "purchase_count": [15, 20, 8, 12, 25],
    "region_code": [1, 2, 1, 3, 2],
    "monthly_income": [400, 300, np.nan, 350, 500],
    "satisfaction_score": [8.5, 9.0, 7.5, 8.0, np.nan],
}
df = pd.DataFrame(data)

print("--- 1. Original Data with Missing Values ---")
print(df)
print("-" * 60)

features_to_impute = ["age", "purchase_count", "monthly_income", "satisfaction_score"]
data_to_impute = df[features_to_impute].copy()
imputer = KNNImputer(n_neighbors=2)
imputed_values = imputer.fit_transform(data_to_impute)
df_imputed = pd.DataFrame(imputed_values, columns=features_to_impute)
df[features_to_impute] = df_imputed[features_to_impute]

print("--- 2. Data After KNN Imputation (K=2) ---")
print(df)
print("-" * 60)
