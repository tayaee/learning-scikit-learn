"""
Pipeline API
    Allows managing multiple steps, from data preprocessing to model training, as a single object.
    The Pipeline is extremely useful for improving code readability and preventing data leakage that can occur in the preprocessing stage.
    scikit-learn's consistent and rich API helps to easily develop, maintain, and scale machine learning models.

Pipeline:
    A Pipeline is a powerful tool that bundles multiple preprocessing steps and the final model into a single object.
    This demo code integrates data scaling (preprocessing) and model training into a single Pipeline,
    demonstrating how the Pipeline enhances code conciseness and effectively prevents Data Leakage, especially during cross-validation.
"""

# 1. Import necessary libraries
import numpy as np
from sklearn.datasets import load_iris  # Example dataset
from sklearn.linear_model import LogisticRegression  # Final model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline  # Pipeline API
from sklearn.preprocessing import StandardScaler  # Data preprocessing (Scaler)

# 2. Load and split example data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==============================================================================
# Section 1: Without using a Pipeline (Manual Implementation)
# Preprocessing and model training are executed separately.
# ==============================================================================
print("=== [Section 1] Without using a Pipeline ===")

# Preprocessing (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Test data is transformed using the statistics from the training data

# Model Training (LogisticRegression)
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test_scaled)
accuracy_manual = accuracy_score(y_test, y_pred)
print(f"Accuracy with manual implementation: {accuracy_manual:.4f}")
print("-" * 50)


# ==============================================================================
# Section 2: Using a Pipeline
# Preprocessing and model are managed as a single object.
# ==============================================================================
print("=== [Section 2] Using a Pipeline ===")

# Create a Pipeline object
# steps: Consists of a list of tuples: (step name, estimator instance)
# Each step must be a 'Transformer' or a 'final Estimator'.
pipeline = Pipeline([
    ("scaler", StandardScaler()),  # First step: Data Standardization
    (
        "classifier",
        LogisticRegression(random_state=42),
    ),  # Second step: Logistic Regression Model
])

# A single call to pipeline.fit() automates both preprocessing and training.
# Internally, scaler.fit_transform() then classifier.fit() is executed sequentially.
pipeline.fit(X_train, y_train)

# A single call to pipeline.predict() automates both preprocessing and prediction.
# Internally, scaler.transform() then classifier.predict() is executed sequentially.
y_pred_pipeline = pipeline.predict(X_test)
accuracy_pipeline = accuracy_score(y_test, y_pred_pipeline)
print(f"Accuracy using Pipeline: {accuracy_pipeline:.4f}")
print("-" * 50)


# ==============================================================================
# Section 3: Pipeline and Cross-Validation
# Demonstrates how the Pipeline prevents data leakage.
# ==============================================================================
print("=== [Section 3] Pipeline and Cross-Validation ===")

# Pass the Pipeline to the cross_val_score function.
# cross_val_score creates and trains a new Pipeline for each fold.
# Thus, scaling statistics are calculated only with the training data each time.
scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")

print(f"5-Fold Cross-Validation Scores using Pipeline: {scores}")
print(f"Pipeline Cross-Validation Mean Accuracy: {np.mean(scores):.4f}")
print(
    "During cross-validation, the Pipeline performs preprocessing only on the training data of each fold. (Prevents Data Leakage)"
)
print("-" * 50)

"""
=== [Section 1] Without using a Pipeline ===
Accuracy with manual implementation: 1.0000
--------------------------------------------------
=== [Section 2] Using a Pipeline ===
Accuracy using Pipeline: 1.0000
--------------------------------------------------
=== [Section 3] Pipeline and Cross-Validation ===
5-Fold Cross-Validation Scores using Pipeline: [0.96666667 1.          0.93333333 0.9        1.        ]
Pipeline Cross-Validation Mean Accuracy: 0.9600
During cross-validation, the Pipeline performs preprocessing only on the training data of each fold. (Prevents Data Leakage)
"""

"""
Review

Q1. What is the biggest advantage of using a Pipeline? Compare the code in Section 1 and 2 to explain.

    The biggest advantage of the Pipeline is that it **automates the workflow** and **simplifies the code** by integrating the preprocessing and model training steps into a single object.

    Section 1 (Manual Implementation):
        Requires separate declaration of StandardScaler and LogisticRegression, and sequential calls to fit_transform(), transform(), fit(), and predict(). This complicates the code.

    Section 2 (Pipeline):
        By creating just one Pipeline object and calling its fit() and predict() methods, all preprocessing and training processes are handled automatically internally. This greatly **improves code readability and maintainability**.

Q2. Explain the internal sequence of operations when `pipeline.fit(X_train, y_train)` is called.

    When `pipeline.fit()` is called, the Pipeline executes each step in the order defined in `steps`.

        **First Step (scaler)**:
            The `fit_transform(X_train)` method of the StandardScaler object is called. It **calculates the mean and standard deviation of the training data (fit)** and **standardizes the data (transform)**. The result is then passed as input to the next step.

        **Second Step (classifier)**:
            The standardized data and `y_train` are received, and the `fit()` method of the LogisticRegression object is called to **train the model**.

Q3. Explain the principle by which the **Pipeline prevents Data Leakage** during the **Cross-Validation** process.

    When a Pipeline is used in cross-validation, such as with `cross_val_score(pipeline, ...)`, a **new Pipeline object is created and trained for each fold**.

    This means that for every fold, `scaler.fit_transform()` is executed **only on the training data**, and only `scaler.transform()` is applied to the test data.

    By ensuring that the test data's information (like mean and standard deviation) is **never used during the training process**, data leakage is effectively prevented, allowing for an objective evaluation of the model's generalization performance.

Q4. In the tuple list passed to the Pipeline's `steps` parameter, such as `('scaler', StandardScaler())`, what are the roles of `'scaler'` and `StandardScaler()` respectively?

    **StandardScaler()** is the **Estimator instance** used for that step in the Pipeline. This instance performs the actual preprocessing or training task.

    **`'scaler'`** is the **name** of that step. This name is used internally by the Pipeline to access the estimator in that step or to specify parameters during **hyperparameter tuning** (e.g., GridSearchCV). For instance, parameters can be specified in the form `scaler__with_mean` in GridSearchCV.
"""
