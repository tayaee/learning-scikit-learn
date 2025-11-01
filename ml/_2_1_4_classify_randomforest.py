"""
Supervised Learning:
    Classification: Used to categorize data into pre-defined classes or categories.
    Examples: sklearn.svm.SVC, sklearn.ensemble.RandomForestClassifier

RandomForestClassifier:
    RandomForestClassifier is a classification model that uses a Random Forest, which is a type of Ensemble Learning technique.
    It builds multiple Decision Trees and aggregates the prediction results of each tree through a majority vote to derive the final result.
    By using multiple trees, it effectively solves the Overfitting problem that a single tree might have,
    and exhibits more stable and higher performance.
"""

# 1. Import necessary libraries
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification  # Generate virtual data for classification
from sklearn.ensemble import RandomForestClassifier  # Random Forest Classification model
from sklearn.metrics import accuracy_score, classification_report  # Model evaluation metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree  # Data splitting

# 2. Generate example data
# Use the make_classification function to generate sample data that mimics a real-world dataset.
# This data has 1000 samples (n_samples), and each sample has 20 features (n_features).
# 10 of these features (n_informative) contain information significant for class classification.
# The data is divided into 2 classes (n_classes), and random_state is set to 42 for reproducible results.
# For example, this data can simulate a scenario where a customer's purchase history (features) is used to predict a purchase decision (class).
#
# Other data generation functions include make_regression, make_blobs, make_moons, make_circles, and make_friedman1.
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)

# 3. Data Splitting
# Split the entire data into training (train) and test sets.
# The training set (80%) is used for model training, and the test set (20%) is used for model performance evaluation.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create Estimator Instance
# Create the RandomForestClassifier model.
# n_estimators: The number of decision trees to include in the forest. A higher number generally improves performance but increases training time.
# random_state: Fixes the model's result to obtain the same outcome upon code re-execution.
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5. Model Training (fit)
# Use the model.fit(X, y) method to train the model on the training data.
# During this process, the model creates and trains multiple decision trees.
print("Starting model training...")
model.fit(X_train, y_train)
print("Model training complete!")

# 6. Prediction (predict)
# Use the model.predict(X) method to perform predictions on the test data using the trained model.
# The prediction results will be class labels, such as 0 or 1.
y_pred = model.predict(X_test)

# 7. Model Evaluation
# Evaluate the model's performance by comparing the actual values (y_test) with the predicted values (y_pred).

# Accuracy: Calculates the ratio of correctly predicted instances out of all predictions.
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Classification Report: Provides more detailed evaluation metrics like Precision, Recall, and F1-score.
# precision: The ratio of actual 'positives' among those predicted as 'positive'.
# recall: The ratio of 'positives' correctly predicted as 'positive' among all actual 'positives'.
# f1-score: The harmonic mean of precision and recall, evaluating the balance between the two metrics.
# support: The number of samples for each class.
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Add Decision Tree Visualization
print("\nStarting Decision Tree visualization...")

# 8-1. Select a Decision Tree to visualize (the first tree in the Random Forest)
print(len(model.estimators_))
tree_to_viz = model.estimators_[0]

# 8-2. Generate Feature Names (since X has 20 features)
feature_names = [f"feature_{i}" for i in range(X.shape[1])]

# 8-3. Set Matplotlib Figure Size
# Set a sufficiently large size for the tree to be clearly visible.
plt.figure(figsize=(20, 10))

# 8-4. Visualize using plot_tree
plot_tree(
    tree_to_viz,
    feature_names=feature_names,  # Feature names
    class_names=["Class 0", "Class 1"],  # Class names
    filled=True,  # Color-code to distinguish classes
    rounded=True,  # Round the node corners
    proportion=False,  # Display sample counts instead of proportions
    max_depth=3,  # Limit tree depth to 3 to reduce complexity (optional)
)

plt.title("Visualization of the First Decision Tree in the Random Forest (Depth limited to 3)")
plt.show()

print("Decision Tree visualization complete!")

"""
Model training started...
Model training complete!

Model Accuracy: 0.92

Classification Report:
                      precision    recall  f1-score   support
                            [2]       [3]       [4]       [5]

           0               0.95      0.89      0.92       104
           1               0.89      0.95      0.92        96

    accuracy                                0.92       200 [1]
   macro avg               0.92      0.92      0.92       200 [6]
weighted avg               0.92      0.92      0.92       200 [7]

How to Interpret the Report:

[1] "accuracy 0.92 200" means that 92% of the total 200 test samples were correctly classified.
    This is the overall result aggregating precision of 0.95 and 0.89, indicating the model achieved an overall accuracy of 92%.

Detailed Metrics by Class (0, 1):

[2] precision: 0.95 for Class 0 and 0.89 for Class 1.
    This means that 95% of the samples predicted as 'positive' for Class 0 were actually 'positive', and 89% for Class 1.

[3] recall: 0.89 for Class 0 and 0.95 for Class 1.
    This means that among all actually 'positive' samples, the model correctly predicted 89% for Class 0 and 95% for Class 1.

[4] f1-score: 0.92 for Class 0 and 0.92 for Class 1.
    As the harmonic mean of precision and recall, this shows balanced performance for both classes. If one metric is low, the f1-score will be low.
    When both are harmoniously high, the f1-score is also high.

[5] support: The actual number of samples in each class, with 104 for Class 0 and 96 for Class 1.
    This indicates how balanced the dataset is in terms of class representation.

[6] macro avg: A simple average of the class metrics (precision, recall, f1-score), excluding support, considering each class's performance equally.

[7] weighted avg: A weighted average considering the support of each class, where metrics are adjusted by the class sample size.
    weighted_precision = (precision_0 * support_0 + precision_1 * support_1) / (support_0 + support_1)
    weighted_recall = (recall_0 * support_0 + recall_1 * support_1) / (support_0 + support_1)
    weighted_f1-score = (f1-score_0 * support_0 + f1-score_1 * support_1) / (support_0 + support_1)

Overall Interpretation Summary:

1. Evaluate Overall Performance: Check **accuracy** to understand the general prediction capability.

2. Check Class Imbalance: Check **support** to verify the number of samples in each class and understand the dataset's balance. In this example, 104 and 96 indicate a relatively balanced state.

3. Detailed Class Performance Analysis: Analyze **precision, recall, and f1-score** for each class to identify the model's strengths and weaknesses.

For Class 0, 0.95 > 0.89, meaning the model is very cautious and accurate 'when it says a sample is Class 0'.
For Class 1, 0.89 < 0.95, meaning the model is good at 'not missing (finding) samples that are truly Class 1'.

4. Evaluate Balance: The f1-scores are both very high at 0.92, indicating balanced performance across the board.

"""

"""
Review

Q1. How does RandomForestClassifier perform classification?

    - RandomForestClassifier uses an **Ensemble Learning** technique.
      It builds multiple independent Decision Trees, aggregates the predictions made by each tree, and determines the final classification result through a majority vote.
      This approach mitigates the drawback of a single Decision Tree (Overfitting) and increases the model's stability and accuracy.

Q2. What is the role of the n_estimators parameter? What are the advantages and disadvantages of increasing its value?

    - **n_estimators** is the parameter that specifies the number of Decision Trees making up the RandomForestClassifier model.

    - Advantages: Increasing n_estimators means more trees participate in prediction, which generally leads to higher model accuracy and improved generalization performance.

    - Disadvantages: As the number of trees increases, model training time lengthens and memory usage increases.

Q3. Explain the roles and the order of use for the fit() and predict() methods in the code.

    - The **fit()** method trains the model using the X_train and y_train data.
      During this process, the model learns the data patterns and finds the optimal model parameters.

    - The **predict()** method performs predictions by feeding new data (X_test) into the trained model.

    - These two methods must be used in the order: **fit()** -> **predict()**.
      A model cannot make predictions without being trained first.

Q4. Why is using classification_report beneficial for model evaluation, in addition to accuracy_score?

    - **accuracy_score** only reports the ratio of correct predictions out of the total, which can be misleading if the data classes are imbalanced (e.g., if one class has significantly more data than the others).

    - **classification_report** provides detailed, per-class performance metrics such as **Precision**, **Recall**, and **F1-score**.
      This allows for a deeper analysis, revealing how well the model predicts specific classes and which classes it struggles with.
"""
