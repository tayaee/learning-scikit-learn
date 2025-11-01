import pandas as pd

# 1. Data Preparation (Assuming a simple regression problem)
data = {
    "Feature_X": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Actual_Y": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55],  # The true labels (target variable)
}
df = pd.DataFrame(data)

# Initial Prediction (XGBoost typically starts with the average of the target variable)
initial_prediction = df["Actual_Y"].mean()
df["F0_Prediction"] = initial_prediction

print("--- 1. Initial Prediction (F0) ---")
print(f"Initial Prediction (Mean): {initial_prediction:.2f}\n")


# === 2. First Model (M1) Training ===
# M1's objective is to predict the 'R1_Residual' (the error from F0).

# 2-1. Calculate the Residual: R1 = Actual_Y - F0_Prediction
df["R1_Residual"] = df["Actual_Y"] - df["F0_Prediction"]

# Simulate M1 training: Assume M1 is trained to predict R1_Residual.
# (We simulate M1 capturing about 60% of the R1 residual)
df["M1_Output"] = df["R1_Residual"] * 0.6

# 2-2. Update the Final Prediction: F1 = F0 + M1
df["F1_Prediction"] = df["F0_Prediction"] + df["M1_Output"]

print("--- 2. First Model (M1) Training Result ---")
print(df[["Actual_Y", "R1_Residual", "M1_Output", "F1_Prediction"]].head(3))
print("M1 predicted the residual R1 and corrected the initial prediction.\n")


# === 3. Second Model (M2) Training ===
# M2's objective is to predict the remaining residual (R2) from F1.

# 3-1. Calculate the New Residual: R2 = Actual_Y - F1_Prediction
df["R2_Residual"] = df["Actual_Y"] - df["F1_Prediction"]

# Simulate M2 training: Assume M2 is trained to predict R2_Residual.
# (We simulate M2 capturing about 80% of the remaining R2 residual)
df["M2_Output"] = df["R2_Residual"] * 0.8

# 3-2. Update the Final Prediction: F2 = F1 + M2
df["F2_Prediction"] = df["F1_Prediction"] + df["M2_Output"]

print("--- 3. Second Model (M2) Training Result ---")
print(df[["Actual_Y", "R2_Residual", "M2_Output", "F2_Prediction"]].head(3))
print(f"\nMean of the Final Residual (R3 if we continued): {df['R2_Residual'].mean():.4f}")
print("M2 predicted the remaining residual R2, further correcting the F1 prediction.")
