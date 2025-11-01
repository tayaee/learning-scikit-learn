"""
Model Selection and Evaluation Tools
    Provides APIs to help evaluate model performance and find optimal hyperparameters.
        sklearn.model_selection: Used to split data into training and testing sets or
            perform cross-validation (cross_val_score, KFold).
        sklearn.metrics: Offers various metrics for evaluating model performance, such as
            accuracy (accuracy_score), precision (precision_score), recall (recall_score), and F1 score (f1_score).

Understanding the Confusion Matrix:
    These four metrics are calculated by combining the four values of the Confusion Matrix (True Positive, False Positive, False Negative, True Negative).
    Therefore, by first understanding the Confusion Matrix, the formula and meaning of each metric can be easily grasped.

Trade-off Between Metrics:
    Precision and Recall are often in a trade-off relationship.
    That is, an increase in one tends to lead to a decrease in the other.
    Without understanding this relationship, looking at only one metric can lead to a misunderstanding of the model's true performance.
    For example, a model that always predicts 'Positive' will have high recall but very low precision.
    The F1 score is a metric that balances this precision and recall, so understanding all three together is necessary for clear meaning.

Necessity of Diverse Evaluation Criteria:
    **Accuracy** is the most intuitive but has a limitation in that it may not properly evaluate the model's performance in cases of **Class Imbalance**.
    For instance, in a dataset where 99 out of 100 samples are the Negative class, a model that always predicts 'Negative' will have 99% accuracy, but it is actually a meaningless model that learned nothing.
    This is where Precision, Recall, and F1 Score play an important role. Observing these metrics together helps determine how well the model predicts the minority class.

Recommended Learning Sequence
    1. Completely understand the concept of the four components (TP, FP, FN, TN) of the **Confusion Matrix**.
    2. Check the formulas for how accuracy_score, precision_score, recall_score, and f1_score are calculated using the values from the Confusion Matrix.
    3. Learn the meaning of each metric and in which situations each is an important indicator through examples.
        Precision is important when minimizing 'False Positives (FP)' is critical (e.g., spam email classification).
        Recall is important when not missing 'True Positives (TP)' is critical (e.g., cancer diagnosis).
        F1 Score is important when both precision and recall are critical.
    4. Finally, practice calculating and interpreting these four metrics together in actual code.

This demo code integrates the main classification evaluation metrics from the sklearn.metrics module.
It is structured to clearly convey the meaning and use cases of each metric, especially based on the concept of the **Confusion Matrix**.
"""

# 1. Import necessary libraries
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# 2. Generate example data
# Generate data with class imbalance. (Class 0: 900 samples, Class 1: 100 samples)
# This dataset helps illustrate the limitations of accuracy.
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=2,
    n_informative=10,
    n_redundant=0,
    weights=[0.9, 0.1],  # Set class imbalance
    random_state=42,
)

# 3. Data Split and Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 4. Confusion Matrix Generation and Analysis
# The Confusion Matrix is a table that shows how well the model's predictions match the actual correct answers.
# Confusion Matrix Structure:
# [TN, FP]
# [FN, TP]
# - TP (True Positive): Actual label is Positive and prediction is Positive (Correctly identified)
# - FN (False Negative): Actual label is Positive but prediction is Negative (Incorrect, Missed)
# - FP (False Positive): Actual label is Negative but prediction is Positive (Incorrect, Falsely predicted)
# - TN (True Negative): Actual label is Negative and prediction is Negative (Correctly identified)
conf_matrix = confusion_matrix(y_test, y_pred)
print("=== Confusion Matrix ===")
print(conf_matrix)
print(f"TP (True Positive): {conf_matrix[1, 1]}")
print(f"FN (False Negative): {conf_matrix[1, 0]}")
print(f"FP (False Positive): {conf_matrix[0, 1]}")
print(f"TN (True Negative): {conf_matrix[0, 0]}")
print("-" * 50)


# 5. Calculation and Interpretation of Various Evaluation Metrics
# These metrics are calculated by combining the values of the Confusion Matrix.

# 5-1. Accuracy
# The ratio of correctly predicted instances out of all predictions. (TP + TN) / (TP + FN + FP + TN)
# Accuracy can distort model performance when there is class imbalance.
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# 5-2. Precision
# The ratio of actual 'Positive' instances among all instances the model predicted as 'Positive'. TP / (TP + FP)
# Used in situations where minimizing False Positives (FP) is critical (e.g., spam email filtering, cryptocurrency fraud detection).
precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.4f}")

# 5-3. Recall
# The ratio of 'Positive' instances that the model correctly predicted as 'Positive' among all actual 'Positive' instances. TP / (TP + FN)
# Used in situations where minimizing False Negatives (FN) is critical (e.g., cancer diagnosis, disaster prediction).
recall = recall_score(y_test, y_pred)
print(f"Recall: {recall:.4f}")

# 5-4. F1 Score
# The harmonic mean of Precision and Recall. 2 * (Precision * Recall) / (Precision + Recall)
# Achieves a high value when both Precision and Recall are balanced and high. Used when both metrics are important.
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.4f}")
print("-" * 50)

"""
Execution Example

=== Confusion Matrix ===
[[262   6]
 [ 22  10]]
TP (True Positive): 10
FN (False Negative): 22
FP (False Positive): 6
TN (True Negative): 262
--------------------------------------------------
Accuracy: 0.9067
Precision: 0.6250
Recall: 0.3125
F1 Score: 0.4167
--------------------------------------------------
"""

"""
Review

Q1. Why can't we say the model is perfect, even though the accuracy_score came out high in the demo code?

    The demo code used a class imbalanced dataset, where 90% of the total data belongs to Class 0 and 10% to Class 1.

    If the model were to predict everything as 'Class 0', the accuracy would still be close to 90%.

    This illustrates the limitation of accuracy in class imbalanced scenarios, as it fails to properly reflect the model's true performance.
    A high accuracy can be achieved even if the model has no predictive ability for the minority class (Class 1).

Q2. Explain the definitions of Precision and Recall using the components of the Confusion Matrix (TP, FP, FN, TN), and provide real-world examples where each is important.

    Precision: TP / (TP + FP)
        The ratio of instances where the prediction was "Positive" that were actually "Positive". Used when minimizing False Positives (FP) is critical.
        Example: Spam email filtering. A high precision is required because a non-spam, important email (Negative) must not be incorrectly classified as spam (Positive - FP).
    Recall: TP / (TP + FN)
        The ratio of actual "Positive" instances that the model correctly predicted as "Positive". Used when minimizing False Negatives (FN) is critical.
        Example: Cancer diagnosis. Since diagnosing an actual cancer patient (Positive) as non-cancer (Negative - FN) is fatal, finding all cancer patients (high recall) is critical.

Q3. Why is the F1 Score calculated as the 'harmonic mean' of precision and recall? Why not use the simple arithmetic mean?

    The arithmetic mean can be skewed by extreme values.
    For example, with 100% precision and 0% recall, the arithmetic mean is 50%, but this model is practically meaningless as it made no useful predictions.

    The harmonic mean heavily penalizes the overall score when one of the metrics is close to zero.
    The F1 Score is designed to place a greater weight on the lower of the two metrics (precision and recall), ensuring that a high score is achieved only when both metrics are balanced and high.

    The formula for calculating the harmonic mean is:

        F1 = 2 * (Precision * Recall) / (Precision + Recall)


Q4. Why is it important to look at other metrics alongside accuracy_score in class imbalanced data?

    In data with class imbalance, the accuracy_score can be high even if the model only predicts the majority class.
    This leads to the misconception that the model is performing well, even though it may have completely failed to learn the minority class.

    **Precision, Recall, and F1 Score** provide a more detailed view of the model's predictive ability for the minority class.
    Specifically, a low **Recall** for the minority class clearly indicates that the model is failing to identify samples belonging to that class.
    Therefore, these metrics must be considered together to accurately evaluate the model's true performance.

    One False Positive is critical $\rightarrow$ Spam Filter $\rightarrow$ Select a high **Precision** model.
    Missing one True Positive is critical $\rightarrow$ Cancer Diagnosis $\rightarrow$ Select a high **Recall** model.
"""
