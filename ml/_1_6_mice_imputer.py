"""
Summary of MICE Imputer Working Principle

MICE (Multiple Imputation by Chained Equations) repeatedly fills missing values in the following sequence:
Initial Imputation:
    All missing values are temporarily filled using a SimpleImputer (mean/median, etc.).
Iteration:
    The first field with missing values (monthly_income) is selected. A regression model is built using this field as the **dependent variable (Y)**, and all other fields (age, purchase_count, satisfaction_score, etc.) as the **independent variables (X)**.
    This model is used to predict and substitute the missing values in monthly_income.
    The second field with missing values (satisfaction_score) is selected, and the missing values are predicted and substituted using the remaining fields (including the newly imputed monthly_income) as independent variables.
    This process is repeated for a **specified number of times (max_iter)**. The predicted values tend to converge more stably with each iteration.
"""

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

data = {
    "customer_id": [1, 2, 3, 4, 5],
    "age": [35, 28, 45, 32, 50],
    "purchase_count": [15, 20, 8, 12, 25],
    "monthly_income": [400, 300, np.nan, 350, 500],
    "satisfaction_score": [8.5, 9.0, 7.5, 8.0, np.nan],
}
df = pd.DataFrame(data)

print("--- 1. Original Data with Missing Values ---")
print(df)
print("-" * 60)

features_to_impute = ["age", "purchase_count", "monthly_income", "satisfaction_score"]
data_to_impute = df[features_to_impute].copy()
# IterativeImputer implements the MICE approach in scikit-learn.
# estimator=BayesianRidge() specifies the regression model used for imputation within the iterations.
# max_iter=10 sets the number of imputation cycles.
imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
imputed_values = imputer.fit_transform(data_to_impute)
df_imputed = pd.DataFrame(imputed_values, columns=features_to_impute)
df[features_to_impute] = df_imputed[features_to_impute]

print("--- 2. Data After MICE Imputation (IterativeImputer) ---")
print(df)
print("-" * 60)
