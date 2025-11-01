"""
DecisionTreeClassifier:
    One of the classification algorithms in Supervised Learning.
    It forms a 'Tree' structure by creating questions based on data features to partition the data.
    It uses metrics like Information Gain or Gini Impurity to select the most efficient feature and criterion
    to split the data and separate classes.
    A single tree is easy to interpret but has a high risk of Overfitting the data.
"""

# 1. Import required libraries
from sklearn.datasets import make_classification  # Generate virtual data for classification
from sklearn.metrics import accuracy_score, classification_report  # Model evaluation metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import (
    DecisionTreeClassifier,  # ✨ Decision Tree Classification Model ✨,  # Function to visualize the decision tree
)

from ml.ut_decision_tree import visualize_decision_tree


# ----------------------------------------------------------------------

# 2. Generate Example Data
# Using a virtual classification dataset
# 1000 samples, 20 features (10 of which are informative), 2 classes
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)

# 3. Data Splitting
# Split into Training (80%) and Testing (20%) sets for model training and evaluation.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create Estimator Instance
# Instantiate the DecisionTreeClassifier model.
# max_depth: Limits the maximum depth of the tree. Deeper trees are more prone to overfitting.
# criterion: Specifies the splitting criterion (impurity measure). Use 'gini' (Gini Impurity) or 'entropy' (Information Gain).
# random_state: Fixes the result for reproducibility.
model = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=42)

# 5. Model Training (fit)
# Fit the Decision Tree to the training data using the model.fit(X, y) method.
print("Starting Decision Tree model training...")
model.fit(X_train, y_train)
print("Model training complete!")

# 6. Prediction (predict)
# Perform class predictions on the test data using the trained model.
y_pred = model.predict(X_test)

# 7. Model Evaluation
# Evaluate performance by comparing true values and predicted values.

# Accuracy: The ratio of correct predictions out of all predictions.
accuracy = accuracy_score(y_test, y_pred)
print(f"\nDecision Tree Model Accuracy: {accuracy:.2f}")

# Classification Report: Provides detailed metrics such as Precision, Recall, F1-Score.
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Decision Tree Visualization
# Unlike RandomForest, the entire Decision Tree model is a single tree, so we visualize it directly.
print("\nStarting Decision Tree visualization...")

# Generate feature names (since X has 20 features) and class names
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
class_names = ["Class 0", "Class 1"]

# Call the extracted function
visualize_decision_tree(model, feature_names, class_names)

print("Decision Tree visualization complete!")
