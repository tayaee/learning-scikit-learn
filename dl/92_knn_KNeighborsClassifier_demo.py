import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

# 1. Load and Split Data
# -----------------------------------------------
# Load the Iris dataset. (Features: sepal length/width, petal length/width | Target: Iris species)
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into Training and Testing sets.
# test_size=0.3: Use 30% of the total data for testing.
# random_state=42: Fix the random seed for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Training Data Size: {len(X_train)} samples")  # Represents the 'Sample_Size' role in the KNN model
print(f"Test Data Size: {len(X_test)} samples")


# 2. Build and 'Train' the Model
# ----------------------------------------------------
# Create the KNN Classifier model.
# n_neighbors=5: Set K=5, meaning it will reference the 5 nearest neighbors.
knn_model = KNeighborsClassifier(n_neighbors=5)

# **KNN's 'Training'**: Simply stores the entire training dataset (X_train, y_train).
print("\n--- Starting KNN Model Build ---")
knn_model.fit(X_train, y_train)
print("--- KNN Model Build Complete (Training Data Stored) ---")


# 3. Predict and Evaluate the Model
# ---------------------------------------------
# Use the trained model to make predictions on the unseen test data (X_test).
y_pred = knn_model.predict(X_test)

# B. Print Detailed Performance Report
print("\nDetailed Classification Report:")
cm = classification_report(y_test, y_pred, target_names=iris.target_names)
print(cm)

# A. Evaluate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\n[Model Performance Evaluation - K=5]")
print(f"Accuracy: {accuracy:.4f}")

f1_score = f1_score(y_test, y_pred, average="weighted")
print(f"F1 Score: {f1_score:.4f}")

precision = precision_score(y_test, y_pred, average="weighted")
print(f"Precision: {precision:.4f}")

recall = recall_score(y_test, y_pred, average="weighted")
print(f"Recall: {recall:.4f}")
