import numpy as np
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from ut_decision_tree import visualize_decision_tree

# 1. Data Loading and Preprocessing
# Load the iris dataset from seaborn
iris = sns.load_dataset("iris")

# Separate features (X) and target (y)
X = iris.drop("species", axis=1)
y = iris["species"]

# Encode the target variable (string) into numbers (0, 1, 2) for the decision tree
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# define the parameters of the tree to iterate over
max_depth_values = np.arange(2, 11, 2)
max_leaf_nodes_values = np.arange(10, 51, 10)
min_samples_split_values = np.arange(10, 51, 10)

# initialize variables to store the best model and its performance
best_estimator = None
best_score_diff = float("inf")

# iterate over all combinations of the specified parameter values
print("--- Manual Loop Start: Finding the most 'balanced' model ---")
for max_depth in max_depth_values:
    for max_leaf_nodes in max_leaf_nodes_values:
        for min_samples_split_value in min_samples_split_values:
            # initialize the tree with the current set of parameters
            estimator = DecisionTreeClassifier(
                max_depth=max_depth,
                max_leaf_nodes=max_leaf_nodes,
                min_samples_split=min_samples_split_value,
                random_state=42,
            )

            # fit the model to the training data
            estimator.fit(X_train, y_train)

            # make predictions on the training and test sets
            y_train_pred = estimator.predict(X_train)
            y_test_pred = estimator.predict(X_test)

            # calculate F1 scores for training and test sets (using average='macro')
            train_f1_score = f1_score(y_train, y_train_pred, average="macro")
            test_f1_score = f1_score(y_test, y_test_pred, average="macro")

            # calculate the absolute difference between training and test F1 scores
            score_diff = abs(train_f1_score - test_f1_score)

            # update the best estimator and best score if the current one has a smaller score difference
            if score_diff < best_score_diff:
                # Print a message indicating a new well-balanced model has been found
                print(
                    f"New well-balanced model: Diff={score_diff:.4f}"
                    f", Test F1={test_f1_score:.4f}"
                    f", Params: max_depth={max_depth}"
                    f", max_leaf_nodes={max_leaf_nodes}"
                    f", min_samples_split={min_samples_split_value}"
                )
                best_score_diff = score_diff
                best_estimator = estimator

print(f"\nBest Balanced Estimator (Manual): {best_estimator}")
print(f"Best Balanced Score Difference: {best_score_diff:.4f}")

feature_names = X.columns.tolist()
class_names = le.classes_.tolist()
visualize_decision_tree(best_estimator, feature_names, class_names)
pass
