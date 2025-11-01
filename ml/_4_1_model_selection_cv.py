"""
Model Selection and Evaluation Tools
    Provides APIs to help evaluate model performance and find optimal hyperparameters.
    sklearn.model_selection: Used to split data into training and testing sets or perform cross-validation (cross_val_score, KFold).
    sklearn.metrics: Provides various metrics for evaluating model performance. Includes accuracy (accuracy_score), precision (precision_score), recall (recall_score), and F1 score (f1_score).

train_test_split
    The most fundamental concept. Essential for understanding the basic train-evaluate process, the simplest form of model validation.
    One must first learn why data is split into a Training set and a Test set, and the concept of using the test set to assess a model's generalization performance.
    Learning Goal: To answer: "Why must data be split?" and "How is Overfitting evaluated?"

KFold
    A method to overcome the limitations of train_test_split. A single split can bias the model toward a specific subset of the data.
    KFold is crucial for understanding the basic principle of **Cross-Validation**, where data is divided into multiple Folds, and each fold is cyclically used as the test set to train and evaluate the model multiple times.
    Learning Goal: To answer: "What are the problems with a single split?" and "Why is cross-validation more reliable?"

cross_val_score
    A convenient function that automates the principle of KFold. It simplifies the complex process of manually creating a KFold object and running a loop into a single function call.
    By learning this function, one learns how to efficiently apply cross-validation in actual code.
    Learning Goal: To answer: "How is cross-validation run easily?" and "How are multiple evaluation scores interpreted?"

This demo code sequentially demonstrates the model validation process using train_test_split, KFold, and cross_val_score.
Each code block clearly shows the difference in how model performance is evaluated, with comments maximizing the educational effect.
"""

# 1. Import necessary libraries
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression  # Classification model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

# 2. Generate example data
# Create virtual data for classification with 1000 samples and 2 classes.
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# 3. Create Model Instance
# Use the Logistic Regression model.
model = LogisticRegression(random_state=42)

# ==============================================================================
# Section 1: Single Model Evaluation using train_test_split
# The most basic method of model validation. Training and evaluation are done with a single data split.
# ==============================================================================

print("=== [Section 1] Single Model Evaluation using train_test_split ===")

# Split the data into a train set and a test set only once.
# test_size=0.3: 30% of the total data is used for the test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model with the training set.
model.fit(X_train, y_train)

# Perform prediction with the test set.
y_pred = model.predict(X_test)

# Evaluate the accuracy of the prediction results.
single_accuracy = accuracy_score(y_test, y_pred)
print(f"Single Split Accuracy: {single_accuracy:.4f}")

# Drawback: The result can vary depending on the data distribution of the single test set, leading to low reliability.
print("-" * 50)


# ==============================================================================
# Section 2: Cross-Validation using KFold (Manual Implementation)
# Evaluate the model multiple times by splitting the data into K folds to reduce performance variation due to data bias.
# ==============================================================================

print("=== [Section 2] Cross-Validation using KFold (Manual Implementation) ===")

# Create the KFold object.
# n_splits=5: Splits the data into 5 folds.
# shuffle=True: Shuffles the data before splitting.
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

accuracies_kfold = []

# Perform cross-validation by looping through the KFold object.
# kfold.split(X) generates train/test indices.
# KFold repeats the train-evaluate process 5 times with these 5 folds. Each iteration proceeds as follows:
#   1st Iteration:
#       Training data: Folds 1, 2, 3, 4 (Total 800 samples)
#       Evaluation data: Fold 5 (Total 200 samples)
#       Result: Record the 1st accuracy score
#   2nd Iteration:
#       Training data: Folds 1, 2, 3, 5 (Total 800 samples)
#       Evaluation data: Fold 4 (Total 200 samples)
#       Result: Record the 2nd accuracy score
#   3rd Iteration:
#       Training data: Folds 1, 2, 4, 5 (Total 800 samples)
#       Evaluation data: Fold 3 (Total 200 samples)
#       Result: Record the 3rd accuracy score
#   4th Iteration:
#       Training data: Folds 1, 3, 4, 5 (Total 800 samples)
#       Evaluation data: Fold 2 (Total 200 samples)
#       Result: Record the 4th accuracy score
#   5th Iteration:
#       Training data: Folds 2, 3, 4, 5 (Total 800 samples)
#       Evaluation data: Fold 1 (Total 200 samples)
#       Result: Record the 5th accuracy score
# After these 5 iterations, we obtain a total of 5 accuracy scores.
# The final result of KFold is the average of these 5 scores, which is the 'Cross-Validation Mean Accuracy'.
for train_index, test_index in kfold.split(X):
    # Extract training/test data corresponding to each fold
    X_train_k, X_test_k = X[train_index], X[test_index]
    y_train_k, y_test_k = y[train_index], y[test_index]

    # Re-initialize and train the model each time.
    model_k = LogisticRegression(random_state=42)
    model_k.fit(X_train_k, y_train_k)

    # Evaluate and save the accuracy
    y_pred_k = model_k.predict(X_test_k)
    accuracies_kfold.append(accuracy_score(y_test_k, y_pred_k))

print(f"5-Fold Cross-Validation Scores: {accuracies_kfold}")
print(f"5-Fold Cross-Validation Mean Accuracy: {np.mean(accuracies_kfold):.4f}")
print("-" * 50)


# ==============================================================================
# Section 3: Cross-Validation using cross_val_score (Automated)
# Execute the complex loop in Section 2 easily in a single line.
# ==============================================================================

print("=== [Section 3] Cross-Validation using cross_val_score (Automated) ===")

# Use the cross_val_score function.
# Arguments: (Model, Entire Feature Data, Entire Target Data, Number of CV Folds, Evaluation Metric)
scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

print(f"5-Fold Cross-Validation Scores obtained with cross_val_score: {scores}")
print(f"cross_val_score Mean Accuracy: {np.mean(scores):.4f}")
print("-" * 50)

"""
Execution Example

=== [Section 1] Single Model Evaluation using train_test_split ===
Single Split Accuracy: 0.8500
--------------------------------------------------
=== [Section 2] Cross-Validation using KFold (Manual Implementation) ===
5-Fold Cross-Validation Scores: [0.855, 0.855, 0.875, 0.865, 0.865]
5-Fold Cross-Validation Mean Accuracy: 0.8630
--------------------------------------------------
=== [Section 3] Cross-Validation using cross_val_score (Automated) ===
5-Fold Cross-Validation Scores obtained with cross_val_score: [0.9   0.885 0.875 0.83  0.845]
cross_val_score Mean Accuracy: 0.8670
--------------------------------------------------
"""

"""
Review

Q1. What is the main limitation of evaluating a model only with train_test_split?

    train_test_split divides the data into a training set and a test set only once.
    Because of this, if the distribution of the split data differs from the overall data, the model's performance can be over- or underestimated.
    This issue is particularly pronounced when the dataset size is small.
    The limitation is that the model's generalization performance is difficult to trust based on a single evaluation result.

Q2. What does the n_splits parameter of the KFold object mean, and how does changing this value affect the cross-validation process?

    **n_splits** refers to the number of folds into which the entire data is divided for cross-validation.
    For example, setting n_splits=5 means the data is divided into 5 folds.

    Changing this value affects the **bias-variance trade-off of the model performance estimate itself**:
        If n_splits is large (e.g., 10-fold):
            - **Decreases Evaluation Bias**: Since each model is trained with more data, the evaluation score is closer to the 'actual performance when trained with all data'. (Positive)
            - **Increases Evaluation Variance**: However, the training data for each fold becomes very similar (with significant overlap), leading to models that are trained very similarly to each other.
            This can cause the evaluation score to jump or drop significantly depending on the data split method. In other words, the **stability of the evaluation result decreases.**
        If n_splits is small (e.g., 3-fold):
            - **Increases Evaluation Bias**: Since each model is trained with less data, the evaluation score may be more pessimistic (lower) than the 'actual performance'.
            - **Decreases Evaluation Variance**: The training data for each fold is more different from one another, so the scores do not change significantly even after multiple evaluations. In other words, the **stability of the evaluation result increases.**

    Conclusion: k=5 or k=10 are widely used as they represent an appropriate balance between the bias and variance of the evaluation.

Q3. What process does cross_val_score internally go through to perform cross-validation? What are the advantages compared to manually implementing KFold?

    **cross_val_score** automatically uses a cross-validation strategy, such as KFold, internally.
    It splits the data according to the number of folds specified in the `cv` argument, iteratively performs training and evaluation, and returns the results as a list.

    Advantage: It handles the tedious steps required for manual KFold implementation—the loop, data indexing, model re-initialization, and score storage—with a single function call, allowing the code to be written much more concisely and efficiently.

Q4. Explain why the mean accuracy scores obtained in Section 2 and Section 3 are similar.

    The two sections yielded similar results because they used the **same cross-validation principle and the same model with identical hyperparameters**.

    Section 2 (manual KFold implementation) and Section 3 (using cross_val_score) both used the same data splitting method with `n_splits=5` and `random_state=42`. Therefore, the internal training and evaluation processes are nearly identical, resulting in similar final mean accuracy scores. This demonstrates that `cross_val_score` effectively encapsulates the functionality of KFold.

Q5. How can the variance (or standard deviation) of the scores obtained from cross-validation be used practically?

    The variance (or standard deviation) of the cross-validation scores is an important indicator of the **stability** of the evaluation result and can be used practically as follows:

    1.  **Comparing Performance Between Models**:
        - Model A: Mean Accuracy 0.92, Standard Deviation 0.01
        - Model B: Mean Accuracy 0.93, Standard Deviation 0.05
        Model B's mean score is slightly higher, but its standard deviation is five times larger.
        This means Model B is an **unstable model** whose performance highly depends on the data split.
        In such a case, choosing the more stable and reliable **Model A**, even with a slightly lower average performance, might be the wiser decision.

    2.  **Estimating the Confidence Interval**:
        The mean score and standard deviation can be used together to roughly estimate the confidence interval for the model's performance.
        A common method is to calculate `Mean Score ± 2 * Standard Deviation`.
        For instance, if the mean accuracy is 0.86 and the standard deviation is 0.01,
        it can be estimated that the model's true performance will likely fall between approximately 84% and 88%.
        This provides much more information than a single score.

    3.  **Hyperparameter Tuning**:
        When using tools like GridSearchCV, it is better to consider combinations with a low `std_test_score` alongside those with the highest `mean_test_score`.
        If the mean scores are similar, selecting the hyperparameter combination with a lower, thus more stable, standard deviation helps **avoid overfitting** and improves generalization performance.

    In conclusion, the variance/standard deviation is maximized in value when it is not interpreted in isolation, but rather **compared with other models or hyperparameter combinations** or used as a **secondary indicator to judge reliability** alongside the mean score.

Q6. Let's estimate the confidence interval for the model's accuracy using the mean and standard deviation from the cross_val_score results.

    The most common method is:

    Performance Interval = Mean Score $\pm$ 2 * Standard Deviation

    The scores 0.9, 0.885, 0.875, 0.83, 0.845 have a Mean = 0.867 and Standard Deviation $\approx$ 0.02885.

    The Performance Interval is $0.867 \pm 2 \times 0.02885 = 0.867 \pm 0.0577$.

    Thus, we can state that the probability of this model's true accuracy being between $[0.8093, 0.9247]$ is approximately 95%.

Q7. What kind of objects can be passed as the first argument (estimator) to cross_val_score?

    The first argument can be **any object that adheres to scikit-learn's Estimator interface**.
    An Estimator refers to any object that learns from data, inherits from `sklearn.base.BaseEstimator`, and possesses a `fit` method.

    Key examples include:
    1.  **Individual Models**:
        Individual classification/regression models such as `LogisticRegression`, `SVC`, `RandomForestClassifier`.
    2.  **Pipelines**:
        A `Pipeline` object, which bundles preprocessing steps and a model into a single estimator, can be passed. This is the best way to perform cross-validation without **data leakage**.
    3.  **Hyperparameter Tuning Objects**:
        `GridSearchCV` and `RandomizedSearchCV` objects, once training is complete, also function as a single estimator containing the optimal model.

    In conclusion, most objects that have a `fit` method and follow scikit-learn's consistent API can be used with `cross_val_score`.
"""
