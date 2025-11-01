"""
Hyperparameter Tuning
This is the process of finding the optimal combination of hyperparameters to maximize a model's performance. It is a scientific method for finding the best-performing model, going beyond simply running the model once or twice.
    sklearn.model_selection.GridSearchCV: Tests all specified parameter values one by one to find the combination that yields the best performance.
    sklearn.model_selection.RandomizedSearchCV: Selects and tests parameter combinations randomly instead of testing all combinations, which increases efficiency.

GridSearchCV is the most representative method for hyperparameter tuning.
It explores all combinations (the grid) of hyperparameter values specified by the user,
and uses Cross-Validation to find the optimal combination that shows the highest performance.
"""

# 1. Import necessary libraries
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC  # Example Model: SVC (Support Vector Classifier)

# 2. Generate example data
# Virtual dataset for solving a classification problem
X, y = make_classification(n_samples=500, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create Model Instance
# Define the base model that will be the target of hyperparameter tuning.
svc = SVC(random_state=42)

# 4. Define the Hyperparameter Search Grid
# Define the hyperparameters to tune in a dictionary format.
# 'C': Regularization parameter; smaller values mean stronger regularization.
# 'gamma': Coefficient for the kernel function; larger values mean the model is more sensitive to the training data.
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [1, 0.1, 0.01, 0.001],
    "kernel": ["rbf"],  # Use only the 'rbf' kernel in this example
}
print("Hyperparameter search grid to explore:")
print(param_grid)
print("-" * 50)

# 5. Create GridSearchCV Instance
# estimator: The base model to tune
# param_grid: The hyperparameter grid to explore
# cv: The number of cross-validation folds
# scoring: The metric to evaluate model performance (here, 'accuracy')
# verbose: Detail level to show progress (higher is more detailed)
grid_search = GridSearchCV(
    estimator=svc,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    verbose=2,
    n_jobs=-1,  # Use all CPU cores for parallel processing (reduces training time)
)

# 6. GridSearchCV Training (Start Hyperparameter Search)
# Calling the fit() method performs cross-validation for all hyperparameter combinations.
# Total combinations (4 * 4 * 1) * Cross-validation folds (5) = 80 training runs are performed.
print("Starting GridSearchCV hyperparameter search...")
grid_search.fit(X_train, y_train)
print("GridSearchCV hyperparameter search complete!")
print("-" * 50)

# 7. Check Optimal Hyperparameters and Best Performance
# best_params_: The combination of hyperparameters that yielded the best performance through cross-validation.
print(f"Best Hyperparameters: {grid_search.best_params_}")
# best_score_: The average score from cross-validation with the best hyperparameters.
print(f"Best Cross-Validation Score (Accuracy): {grid_search.best_score_:.4f}")

# 8. Final Model Evaluation
# GridSearchCV internally stores the model re-trained with the best hyperparameters.
# Apply this model to the test dataset to evaluate the final performance.
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)
print(f"Final Accuracy on the Test Dataset: {final_accuracy:.4f}")
print("-" * 50)

# Note: Check cross-validation results for all combinations
print("Cross-validation results for all parameter combinations:")
print(grid_search.cv_results_)

"""
Hyperparameter search grid to explore:
{'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
--------------------------------------------------
Starting GridSearchCV hyperparameter search...
Fitting 5 folds for each of 16 candidates, totalling 80 fits
[CV] END .........................C=0.1, gamma=1, kernel=rbf; total time=   0.0s
[CV] END .........................C=0.1, gamma=1, kernel=rbf; total time=   0.0s
[CV] END .........................C=0.1, gamma=1, kernel=rbf; total time=   0.0s
[CV] END ........................C=1, gamma=0.01, kernel=rbf; total time=   0.0s
...
[CV] END .........................C=1, gamma=0.1, kernel=rbf; total time=   0.0s
[CV] END ........................C=1, gamma=0.01, kernel=rbf; total time=   0.0s
[CV] END .........................C=1, gamma=0.1, kernel=rbf; total time=   0.0s
[CV] END ........................C=1, gamma=0.01, kernel=rbf; total time=   0.0s
GridSearchCV hyperparameter search complete!
--------------------------------------------------
Best Hyperparameters: {'C': 0.1, 'gamma': 0.01, 'kernel': 'rbf'}
Best Cross-Validation Score (Accuracy): 0.8725
Final Accuracy on the Test Dataset: 0.9100
--------------------------------------------------
Cross-validation results for all parameter combinations:
{'mean_fit_time': array([0.01839571, 0.01530485, 0.01561785, 0.01471353, 0.01637173,
       0.01195002, 0.00905228, 0.01321878, 0.01988659, 0.01715937,
       0.01059942, 0.01101189, 0.02129307, 0.01780772, 0.0171443 ,
       0.01000037]), 'std_fit_time': array([0.00223373, 0.00075003, 0.00083162, 0.00154257, 0.00117721,
       0.00074147, 0.0013472 , 0.00097681, 0.00108303, 0.00042722,
       0.00049109, 0.00131314, 0.0017248 , 0.00112461, 0.00154393,
       0.00063203]), 'mean_score_time': array([0.01110783, 0.01079998, 0.00982008, 0.00986176, 0.00759864,
       0.00570765, 0.00589366, 0.00951285, 0.01052833, 0.00945263,
       0.00539999, 0.00699944, 0.01048589, 0.00899606, 0.00540047,
       0.00580001]), 'std_score_time': array([0.00066784, 0.00097993, 0.00136087, 0.0007703 , 0.00119786,
       0.00039525, 0.00185585, 0.00045453, 0.00107804, 0.00055787,
       0.00048965, 0.00109489, 0.00117607, 0.00038442, 0.00048945,
       0.00040031]), 'param_C': masked_array(data=[0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0,
                         10.0, 10.0, 100.0, 100.0, 100.0, 100.0],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False],
       fill_value=1e+20), 'param_gamma': masked_array(data=[1.0, 0.1, 0.01, 0.001, 1.0, 0.1, 0.01, 0.001, 1.0, 0.1,
                         0.01, 0.001, 1.0, 0.1, 0.01, 0.001],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False],
       fill_value=1e+20), 'param_kernel': masked_array(data=['rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf',
                         'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf'],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False],
       fill_value=np.str_('?'),
             dtype=object), 'params': [{'C': 0.1, 'gamma': 1, 'kernel': 'rbf'}, {'C': 0.1, 'gamma': 0.1, 'kernel': 'rbf'}, {'C': 0.1, 'gamma': 0.01, 'kernel': 'rbf'}, {'C': 0.1, 'gamma': 0.001, 'kernel': 'rbf'}, {'C': 1, 'gamma': 1, 'kernel': 'rbf'}, {'C': 1, 'gamma': 0.1, 'kernel': 'rbf'}, {'C': 1, 'gamma': 0.01, 'kernel': 'rbf'}, {'C': 1, 'gamma': 0.001, 'kernel': 'rbf'}, {'C': 10, 'gamma': 1, 'kernel': 'rbf'}, {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}, {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}, {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}, {'C': 100, 'gamma': 1, 'kernel': 'rbf'}, {'C': 100, 'gamma': 0.1, 'kernel': 'rbf'}, {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}, {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}], 'split0_test_score': array([0.5   , 0.4875, 0.8625, 0.5   , 0.5   , 0.8375, 0.85  , 0.8875,
       0.5   , 0.85  , 0.8625, 0.85  , 0.5   , 0.85  , 0.825 , 0.85  ]), 'split1_test_score': array([0.5125, 0.5625, 0.85  , 0.5125, 0.5125, 0.85  , 0.8125, 0.85  ,
       0.5125, 0.8375, 0.8125, 0.8375, 0.5125, 0.8375, 0.825 , 0.7625]), 'split2_test_score': array([0.5125, 0.525 , 0.875 , 0.5125, 0.5125, 0.875 , 0.875 , 0.8625,
       0.5125, 0.875 , 0.875 , 0.875 , 0.5125, 0.875 , 0.825 , 0.8875]), 'split3_test_score': array([0.5125, 0.5375, 0.9125, 0.5125, 0.5125, 0.8875, 0.875 , 0.9   ,
       0.5125, 0.875 , 0.8875, 0.875 , 0.5125, 0.875 , 0.85  , 0.875 ]), 'split4_test_score': array([0.5125, 0.525 , 0.8625, 0.5125, 0.5125, 0.8625, 0.9125, 0.8625,
       0.5125, 0.825 , 0.85  , 0.9   , 0.5125, 0.825 , 0.8625, 0.8625]), 'mean_test_score': array([0.51  , 0.5275, 0.8725, 0.51  , 0.51  , 0.8625, 0.865 , 0.8725,
       0.51  , 0.8525, 0.8575, 0.8675, 0.51  , 0.8525, 0.8375, 0.8475]), 'std_test_score': array([0.005     , 0.0242384 , 0.02150581, 0.005     , 0.005     ,
       0.01767767, 0.03297726, 0.01837117, 0.005     , 0.02      ,
       0.02573908, 0.02179449, 0.005     , 0.02      , 0.01581139,
       0.04430011]), 'rank_test_score': array([12, 11,  1, 12, 12,  5,  4,  1, 12,  7,  6,  3, 12,  7, 10,  9],
       dtype=int32)}
"""

"""
Review

Q1. How does GridSearchCV find the optimal hyperparameters? What are the advantages and disadvantages of this method?

    Method: It explores **all possible combinations (the grid)** of hyperparameter values specified by the user one by one.
            For each combination, it performs **Cross-Validation (CV)** for the number of folds specified by `cv`, and selects the combination with the highest average score as the optimal hyperparameter set.

    Advantage: Since all combinations are verified, there is no risk of missing the optimal combination.

    Disadvantage: If the number of hyperparameters or the range of values to explore is large, the **training time can increase exponentially**, making it highly inefficient.

Q2. When the fit() method is called on GridSearchCV, what process occurs internally?

    When `fit()` is called, GridSearchCV performs the following steps:

        1. It creates all hyperparameter combinations defined in `param_grid`.
        2. For each combination, it starts cross-validation by splitting the data into the number of folds specified by `cv`.
        3. In each fold of the cross-validation, the model is trained on the training data and its performance is evaluated on the test data.
        4. The evaluation scores for all folds are averaged to calculate the final score for that combination.
        5. This process is repeated for every hyperparameter combination.
        6. Finally, the combination that yielded the highest average score is stored in `best_params_`, and the final model re-trained on the entire training dataset with this combination is stored in `best_estimator_`.

Q3. In the code, you defined 'C': [0.1, 1, 10, 100] and 'gamma': [1, 0.1, 0.01, 0.001] in the `param_grid` dictionary and set `cv=5`. How many times is the model trained in total?

    The total number of training runs is (Number of C values) * (Number of gamma values) * (Number of kernel values) * (Number of cross-validation folds).
    $4 \times 4 \times 1 \times 5 = 80$
    Therefore, the model is trained a total of **80 times**.

Q4. What is the difference between `grid_search.best_score_` and `final_accuracy`? Why is it important to check both values?

    best_score_:
        This is the **highest average score** obtained by GridSearchCV through cross-validation **within the training dataset**.
        This value is used to evaluate the model's performance during the hyperparameter tuning process.

    final_accuracy:
        This is the score obtained by applying the final model, trained with the optimal hyperparameters found by GridSearchCV, to the **unseen test dataset**.

    Checking both values is important to determine the presence of **Overfitting**. If `best_score_` is high but `final_accuracy` is relatively lower, it's likely that the model has **overfit to the training data**. The `final_accuracy` is the metric that shows the model's true generalization performance.

Q5. What does CV stand for in GridSearchCV, and what is its role?

    CV stands for **Cross-Validation**.

    When GridSearchCV evaluates the performance of each hyperparameter combination, it uses cross-validation instead of evaluating the performance based on a single data split.
    For example, if `cv=5` is set, the training data is divided into 5 folds.
    For one hyperparameter combination, the training and evaluation steps are repeated 5 times in total.
    The average of these 5 evaluation scores is then used as the final performance score for that combination.

    This method helps reduce the randomness associated with data splitting and allows for a more stable and reliable assessment of the model's generalization performance.
    Therefore, CV plays a crucial role in the hyperparameter tuning process.

Q6. How can one check the list of all tuneable hyperparameters for a specific model?

    All scikit-learn estimators possess the **`get_params()`** method.
    Calling this method returns a dictionary containing a list of all hyperparameters that can be set and tuned for that model, along with their current values.

    Example Code:
    ```python
    from sklearn.svm import SVC

    svc = SVC()
    print(svc.get_params())
    ```

    Output Result:
    ```
    {'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'probability': False, 'random_state': None, 'shrinking': True, 'tol': 0.001, 'verbose': False}
    ```
    One can refer to this list when constructing `param_grid` or `param_distributions`.
"""
