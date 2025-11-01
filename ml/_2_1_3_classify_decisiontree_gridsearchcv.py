import numpy as np
import seaborn as sns
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from ut_decision_tree import visualize_decision_tree

# 1. Data Loading and Preprocessing (Same as above)
iris = sns.load_dataset("iris")
X = iris.drop("species", axis=1)
y = iris["species"]
le = LabelEncoder()
y = le.fit_transform(y)

# Since GridSearchCV internally performs cross-validation,
# it is common practice to use the entire data (X, y) to find the optimal model,
# and then finally evaluate the performance on X_test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 2. Define the parameter grid to search
param_grid = {
    "max_depth": np.arange(2, 11, 2),
    "max_leaf_nodes": np.arange(10, 51, 10),
    "min_samples_split": np.arange(10, 51, 10),
}

# 3. Define the DecisionTreeClassifier model
dt = DecisionTreeClassifier(random_state=42)

# 4. Create GridSearchCV object (Set F1 score as the optimization target)
# Use make_scorer to specify that the F1 score should be calculated with 'macro' averaging
f1_scorer = make_scorer(f1_score, average="macro")

# cv=5: Perform 5-fold cross-validation
# n_jobs=-1: Use all available CPU cores for parallel processing
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, scoring=f1_scorer, cv=5, n_jobs=-1)

# 5. Start grid search (Perform 5-fold cross-validation on X_train, y_train)
print("--- GridSearchCV Start: Finding the model with the highest average CV F1 Score ---")
grid_search.fit(X_train, y_train)

# 6. Print the best results
print("\nGridSearchCV Results:")
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Best Cross-Validation F1 Score (Average Test Score): {grid_search.best_score_:.4f}")

# 7. Evaluate performance on the final test dataset using the best model
best_estimator_grid = grid_search.best_estimator_
y_test_pred_grid = best_estimator_grid.predict(X_test)
test_f1_score_final = f1_score(y_test, y_test_pred_grid, average="macro")

print(f"\nFinal Test Set F1 Score (using best model): {test_f1_score_final:.4f}")

feature_names = X.columns.tolist()
class_names = le.classes_.tolist()
# Placeholder for visualization function call (assuming implementation exists elsewhere)
visualize_decision_tree(best_estimator_grid, feature_names, class_names)
pass
