import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# 1. Load and Prepare Data
# Using the well-known Iris dataset for a multi-class classification example (3 classes).
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 2. Initialize and Train the GradientBoostingClassifier Model
# n_estimators: The number of weak learners (trees) to be trained sequentially.
# learning_rate: Controls how much the error from the previous tree is incorporated.
gbrt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

print("--- Starting Gradient Boosting Model Training ---")
# Model training: Internally, the model calculates the residuals (negative gradients)
# and sequentially adds trees to correct errors.
gbrt.fit(X_train, y_train)

print("--- Training Complete ---")

# 3. Prediction and Performance Evaluation
# Perform prediction using the test dataset
y_pred = gbrt.predict(X_test)

# Evaluate Accuracy
accuracy = accuracy_score(y_test, y_pred)

# 4. Output Results
print(f"\nTest Set Accuracy: {accuracy:.4f}")

# 5. Review Internal Boosting Process
# Check the model's performance stage by stage (as each tree is added).
print("\n--- Step-wise Boosting Performance (Accuracy after each tree) ---")

# Calculate performance after each boosting stage
scores = []
for y_pred_stage in gbrt.staged_predict(X_test):
    scores.append(accuracy_score(y_test, y_pred_stage))

# Display accuracy changes for the first 10 stages
print("Accuracy change during the initial 10 stages:")
for i, score in enumerate(scores[:10]):
    print(f"Stage {i + 1}: Accuracy {score:.4f}")
