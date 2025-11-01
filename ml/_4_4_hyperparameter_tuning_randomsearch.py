"""
Hyperparameter Tuning
This is the process of finding the optimal combination of hyperparameters to maximize a model's performance. It is a scientific method for finding the best-performing model, going beyond simply running the model once or twice.
    sklearn.model_selection.GridSearchCV: Tests all specified parameter values one by one to find the combination that yields the best performance.
    sklearn.model_selection.RandomizedSearchCV: Selects and tests parameter combinations randomly instead of testing all combinations, which increases efficiency.

RandomizedSearchCV is a tool for hyperparameter tuning, similar to GridSearchCV.
However, unlike GridSearchCV, which explores all parameter combinations,
it randomly samples parameter combinations a specific number of times based on a user-defined distribution.
The advantage of this method is that it can find the optimal hyperparameters much more efficiently than GridSearchCV, especially when the search space is very large.
"""

# 1. Import necessary libraries
import json
import os
from scipy.stats import uniform  # Library for uniform distribution
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVC

# 2. Generate example data
# Virtual dataset for solving a classification problem
X, y = make_classification(n_samples=500, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create Model Instance
# Define the base model that will be the target of hyperparameter tuning.
svc = SVC(random_state=42)

params = svc.get_params()
print(f"Model params: {json.dumps(params, indent=2)}")


# 4. Define the Hyperparameter Distribution to Explore
# Unlike 'GridSearchCV', you can define a distribution instead of a list of values.
# uniform(loc, scale): Creates a uniform distribution from loc up to loc + scale.
param_distributions = {
    "C": uniform(loc=0.1, scale=100),  # Randomly sampled from a uniform distribution from 0.1 up to 100.1
    "gamma": uniform(loc=0.0001, scale=1),  # Randomly sampled from a uniform distribution from 0.0001 up to 1.0001
    "kernel": ["rbf"],
}
print("Hyperparameter distribution to explore:")
print(param_distributions)
print("-" * 50)

# 5. Create RandomizedSearchCV Instance
# estimator: The base model to tune
# param_distributions: The hyperparameter distribution to explore
# n_iter: The number of random samplings (combinations) to explore
# cv: The number of cross-validation folds
# scoring: The metric to evaluate model performance
# verbose: Detail level to show progress
print(f"os.cpu_count() = {os.cpu_count()}")
random_search = RandomizedSearchCV(
    estimator=svc,
    param_distributions=param_distributions,
    n_iter=20,  # Explore only 20 random combinations
    cv=5,
    scoring="accuracy",
    verbose=2,
    random_state=42,
    n_jobs=-1,  # Use all CPU cores
)

# 6. RandomizedSearchCV Training (Start Hyperparameter Search)
# Since n_iter=20, cross-validation is performed for a total of 20 random combinations.
# Using 5 folds for each combination results in a total of $20 \times 5 = 100$ training runs.
print("Starting RandomizedSearchCV hyperparameter search...")
random_search.fit(X_train, y_train)
print("RandomizedSearchCV hyperparameter search complete!")
print("-" * 50)

# 7. Check Optimal Hyperparameters and Best Performance
# best_params_: The combination of hyperparameters that yielded the best performance through cross-validation.
print(f"Best Hyperparameters: {random_search.best_params_}")
# best_score_: The average score from cross-validation with the best hyperparameters.
print(f"Best Cross-Validation Score (Accuracy): {random_search.best_score_:.4f}")

# 8. Final Model Evaluation
# Apply the model re-trained with the optimal hyperparameters to the test dataset to evaluate final performance.
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)
print(f"Final Accuracy on the Test Dataset: {final_accuracy:.4f}")
print("-" * 50)

# Note: Check cross-validation results for all combinations
print("Cross-validation results for all parameter combinations:")
print(random_search.cv_results_)

"""
Hyperparameter distribution to explore:
{'C': <scipy.stats._distn_infrastructure.rv_continuous_frozen object at 0x000001CCF0F73C10>, 'gamma': <scipy.stats._distn_infrastructure.rv_continuous_frozen object at 0x000001CCF07C8FD0>, 'kernel': ['rbf']}
--------------------------------------------------
Starting RandomizedSearchCV hyperparameter search...
Fitting 5 folds for each of 20 candidates, totalling 100 fits
[CV] END C=37.55401188473625, gamma=0.9508143064099162, kernel=rbf; total time=   0.0s
[CV] END C=37.55401188473625, gamma=0.9508143064099162, kernel=rbf; total time=   0.0s
[CV] END C=37.55401188473625, gamma=0.9508143064099162, kernel=rbf; total time=   0.0s
[CV] END C=83.34426408004217, gamma=0.21243911067827614, kernel=rbf; total time=   0.0s
[CV] END C=83.34426408004217, gamma=0.21243911067827614, kernel=rbf; total time=   0.0s
[CV] END C=83.34426408004217, gamma=0.21243911067827614, kernel=rbf; total time=   0.0s
[CV] END C=18.282496720710064, gamma=0.1835045098534338, kernel=rbf; total time=   0.0s
[CV] END C=18.282496720710064, gamma=0.1835045098534338, kernel=rbf; total time=   0.0s
[CV] END C=18.282496720710064, gamma=0.1835045098534338, kernel=rbf; total time=   0.0s
[CV] END C=18.282496720710064, gamma=0.1835045098534338, kernel=rbf; total time=   0.0s
[CV] END C=73.2993941811405, gamma=0.5987584841970366, kernel=rbf; total time=   0.0s
[CV] END C=18.282496720710064, gamma=0.1835045098534338, kernel=rbf; total time=   0.0s
[CV] END C=30.524224295953772, gamma=0.5248564316322378, kernel=rbf; total time=   0.0s
[CV] END C=73.2993941811405, gamma=0.5987584841970366, kernel=rbf; total time=   0.0s
[CV] END C=30.524224295953772, gamma=0.5248564316322378, kernel=rbf; total time=   0.0s
[CV] END C=30.524224295953772, gamma=0.5248564316322378, kernel=rbf; total time=   0.0s
[CV] END C=73.2993941811405, gamma=0.5987584841970366, kernel=rbf; total time=   0.0s
[CV] END C=30.524224295953772, gamma=0.5248564316322378, kernel=rbf; total time=   0.0s
[CV] END C=30.524224295953772, gamma=0.5248564316322378, kernel=rbf; total time=   0.0s
[CV] END C=43.294501864211576, gamma=0.2913291401980419, kernel=rbf; total time=   0.0s
[CV] END C=43.294501864211576, gamma=0.2913291401980419, kernel=rbf; total time=   0.0s
[CV] END C=43.294501864211576, gamma=0.2913291401980419, kernel=rbf; total time=   0.0s
[CV] END C=43.294501864211576, gamma=0.2913291401980419, kernel=rbf; total time=   0.0s
[CV] END C=43.294501864211576, gamma=0.2913291401980419, kernel=rbf; total time=   0.0s
[CV] END C=61.28528947223795, gamma=0.13959386065204182, kernel=rbf; total time=   0.0s
[CV] END C=61.28528947223795, gamma=0.13959386065204182, kernel=rbf; total time=   0.0s
[CV] END C=61.28528947223795, gamma=0.13959386065204182, kernel=rbf; total time=   0.0s
[CV] END C=61.28528947223795, gamma=0.13959386065204182, kernel=rbf; total time=   0.0s
[CV] END C=15.701864044243651, gamma=0.15609452033620264, kernel=rbf; total time=   0.0s
[CV] END C=61.28528947223795, gamma=0.13959386065204182, kernel=rbf; total time=   0.0s
[CV] END C=29.314464853521816, gamma=0.3664618432936917, kernel=rbf; total time=   0.0s
[CV] END C=29.314464853521816, gamma=0.3664618432936917, kernel=rbf; total time=   0.0s
[CV] END C=29.314464853521816, gamma=0.3664618432936917, kernel=rbf; total time=   0.0s
[CV] END C=29.314464853521816, gamma=0.3664618432936917, kernel=rbf; total time=   0.0s
[CV] END C=29.314464853521816, gamma=0.3664618432936917, kernel=rbf; total time=   0.0s
[CV] END C=45.706998421703595, gamma=0.7852759613930136, kernel=rbf; total time=   0.0s
[CV] END C=45.706998421703595, gamma=0.7852759613930136, kernel=rbf; total time=   0.0s
[CV] END C=45.706998421703595, gamma=0.7852759613930136, kernel=rbf; total time=   0.0s
[CV] END C=45.706998421703595, gamma=0.7852759613930136, kernel=rbf; total time=   0.0s
[CV] END C=45.706998421703595, gamma=0.7852759613930136, kernel=rbf; total time=   0.0s
[CV] END C=20.067378215835976, gamma=0.5143344384136116, kernel=rbf; total time=   0.0s
[CV] END C=20.067378215835976, gamma=0.5143344384136116, kernel=rbf; total time=   0.0s
[CV] END C=20.067378215835976, gamma=0.5143344384136116, kernel=rbf; total time=   0.0s
[CV] END C=20.067378215835976, gamma=0.5143344384136116, kernel=rbf; total time=   0.0s
[CV] END C=59.34145688620425, gamma=0.04655041271999773, kernel=rbf; total time=   0.0s
[CV] END C=20.067378215835976, gamma=0.5143344384136116, kernel=rbf; total time=   0.0s
[CV] END C=5.908361216819946, gamma=0.8662761457749352, kernel=rbf; total time=   0.0s
[CV] END C=59.34145688620425, gamma=0.04655041271999773, kernel=rbf; total time=   0.0s
[CV] END C=59.34145688620425, gamma=0.04655041271999773, kernel=rbf; total time=   0.0s
[CV] END C=59.34145688620425, gamma=0.04655041271999773, kernel=rbf; total time=   0.0s
[CV] END C=59.34145688620425, gamma=0.04655041271999773, kernel=rbf; total time=   0.0s
[CV] END C=60.85448519014384, gamma=0.17062412368729152, kernel=rbf; total time=   0.0s
[CV] END C=15.701864044243651, gamma=0.15609452033620264, kernel=rbf; total time=   0.0s
[CV] END C=5.908361216819946, gamma=0.8662761457749352, kernel=rbf; total time=   0.0s
[CV] END C=60.85448519014384, gamma=0.17062412368729152, kernel=rbf; total time=   0.0s
[CV] END C=60.85448519014384, gamma=0.17062412368729152, kernel=rbf; total time=   0.0s
[CV] END C=60.85448519014384, gamma=0.17062412368729152, kernel=rbf; total time=   0.0s
[CV] END C=60.85448519014384, gamma=0.17062412368729152, kernel=rbf; total time=   0.0s
[CV] END C=6.605159298527951, gamma=0.9489855372533332, kernel=rbf; total time=   0.0s
[CV] END C=6.605159298527951, gamma=0.9489855372533332, kernel=rbf; total time=   0.0s
[CV] END C=6.605159298527951, gamma=0.9489855372533332, kernel=rbf; total time=   0.0s
[CV] END C=6.605159298527951, gamma=0.9489855372533332, kernel=rbf; total time=   0.0s
[CV] END C=96.66320330745593, gamma=0.8084973481164611, kernel=rbf; total time=   0.0s
[CV] END C=6.605159298527951, gamma=0.9489855372533332, kernel=rbf; total time=   0.0s
[CV] END C=96.66320330745593, gamma=0.8084973481164611, kernel=rbf; total time=   0.0s
[CV] END C=5.908361216819946, gamma=0.8662761457749352, kernel=rbf; total time=   0.0s
[CV] END C=96.66320330745593, gamma=0.8084973481164611, kernel=rbf; total time=   0.0s
[CV] END C=96.66320330745593, gamma=0.8084973481164611, kernel=rbf; total time=   0.0s
[CV] END C=96.66320330745593, gamma=0.8084973481164611, kernel=rbf; total time=   0.0s
[CV] END C=30.56137691733707, gamma=0.09777211400638387, kernel=rbf; total time=   0.0s
[CV] END C=30.56137691733707, gamma=0.09777211400638387, kernel=rbf; total time=   0.0s
[CV] END C=30.56137691733707, gamma=0.09777211400638387, kernel=rbf; total time=   0.0s
[CV] END C=30.56137691733707, gamma=0.09777211400638387, kernel=rbf; total time=   0.0s
[CV] END C=30.56137691733707, gamma=0.09777211400638387, kernel=rbf; total time=   0.0s
[CV] END C=68.52330265121569, gamma=0.4402524937396013, kernel=rbf; total time=   0.0s
[CV] END C=68.52330265121569, gamma=0.4402524937396013, kernel=rbf; total time=   0.0s
[CV] END C=68.52330265121569, gamma=0.4402524937396013, kernel=rbf; total time=   0.0s
[CV] END C=37.55401188473625, gamma=0.9508143064099162, kernel=rbf; total time=   0.0s
[CV] END C=68.52330265121569, gamma=0.4402524937396013, kernel=rbf; total time=   0.0s
[CV] END C=68.52330265121569, gamma=0.4402524937396013, kernel=rbf; total time=   0.0s
[CV] END C=37.55401188473625, gamma=0.9508143064099162, kernel=rbf; total time=   0.0s
[CV] END C=60.21150117432088, gamma=0.7081725777960455, kernel=rbf; total time=   0.0s
[CV] END C=60.21150117432088, gamma=0.7081725777960455, kernel=rbf; total time=   0.0s
[CV] END C=73.2993941811405, gamma=0.5987584841970366, kernel=rbf; total time=   0.0s
[CV] END C=73.2993941811405, gamma=0.5987584841970366, kernel=rbf; total time=   0.0s
[CV] END C=15.701864044243651, gamma=0.15609452033620264, kernel=rbf; total time=   0.0s
[CV] END C=2.1584494295802448, gamma=0.9700098521619943, kernel=rbf; total time=   0.0s
[CV] END C=2.1584494295802448, gamma=0.9700098521619943, kernel=rbf; total time=   0.0s
[CV] END C=5.908361216819946, gamma=0.8662761457749352, kernel=rbf; total time=   0.0s
[CV] END C=15.701864044243651, gamma=0.15609452033620264, kernel=rbf; total time=   0.0s
[CV] END C=15.701864044243651, gamma=0.15609452033620264, kernel=rbf; total time=   0.0s
[CV] END C=60.21150117432088, gamma=0.7081725777960455, kernel=rbf; total time=   0.0s
[CV] END C=60.21150117432088, gamma=0.7081725777960455, kernel=rbf; total time=   0.0s
[CV] END C=60.21150117432088, gamma=0.7081725777960455, kernel=rbf; total time=   0.0s
[CV] END C=5.908361216819946, gamma=0.8662761457749352, kernel=rbf; total time=   0.0s
[CV] END C=2.1584494295802448, gamma=0.9700098521619943, kernel=rbf; total time=   0.0s
[CV] END C=83.34426408004217, gamma=0.21243911067827614, kernel=rbf; total time=   0.0s
[CV] END C=2.1584494295802448, gamma=0.9700098521619943, kernel=rbf; total time=   0.0s
[CV] END C=2.1584494295802448, gamma=0.9700098521619943, kernel=rbf; total time=   0.0s
[CV] END C=83.34426408004217, gamma=0.21243911067827614, kernel=rbf; total time=   0.0s
RandomizedSearchCV hyperparameter search complete!
--------------------------------------------------
Best Hyperparameters: {'C': np.float64(59.34145688620425), 'gamma': np.float64(0.04655041271999773), 'kernel': 'rbf'}
Best Cross-Validation Score (Accuracy): 0.8575
Final Accuracy on the Test Dataset: 0.8600
--------------------------------------------------
Cross-validation results for all parameter combinations:
{'mean_fit_time': array([0.02130742, 0.01929746, 0.01660695, 0.02062106, 0.01820703,
       0.01610756, 0.01594458, 0.01764688, 0.01822062, 0.01810708,
       0.01798582, 0.01831441, 0.02009768, 0.0185173 , 0.01587844,
       0.0169672 , 0.02022839, 0.01916642, 0.01709776, 0.01857595]), 'std_fit_time': array([0.00165557, 0.00312994, 0.0023723 , 0.00140278, 0.00221557,
       0.00246028, 0.00183302, 0.0007835 , 0.00114608, 0.00040581,
       0.00086469, 0.00042244, 0.00249609, 0.00093879, 0.00072694,
       0.00095171, 0.00182032, 0.00075235, 0.00049571, 0.00065806]), 'mean_score_time': array([0.01129899, 0.00952578, 0.00927539, 0.00981965, 0.00819988,
       0.00739861, 0.00866036, 0.01078496, 0.0105772 , 0.0101295 ,
       0.01005211, 0.0106389 , 0.01012163, 0.01016927, 0.00716147,
       0.01040177, 0.01112843, 0.01053896, 0.00946593, 0.01058865]), 'std_score_time': array([0.00109023, 0.00181579, 0.00125362, 0.00095619, 0.00074847,
       0.00101675, 0.00210195, 0.00067804, 0.00058698, 0.00126576,
       0.00046465, 0.00029145, 0.00046623, 0.00033546, 0.00048027,
       0.00049318, 0.00081494, 0.00038211, 0.00065887, 0.00048093]), 'param_C': masked_array(data=[37.55401188473625, 73.2993941811405,
                         15.701864044243651, 5.908361216819946,
                         60.21150117432088, 2.1584494295802448,
                         83.34426408004217, 18.282496720710064,
                         30.524224295953772, 43.294501864211576,
                         61.28528947223795, 29.314464853521816,
                         45.706998421703595, 20.067378215835976,
                         59.34145688620425, 60.85448519014384,
                         6.605159298527951, 96.66320330745593,
                         30.56137691733707, 68.52330265121569],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False],
       fill_value=1e+20), 'param_gamma': masked_array(data=[0.9508143064099162, 0.5987584841970366,
                         0.15609452033620264, 0.8662761457749352,
                         0.7081725777960455, 0.9700098521619943,
                         0.21243911067827614, 0.1835045098534338,
                         0.5248564316322378, 0.2913291401980419,
                         0.13959386065204182, 0.3664618432936917,
                         0.7852759613930136, 0.5143344384136116,
                         0.04655041271999773, 0.17062412368729152,
                         0.9489855372533332, 0.8084973481164611,
                         0.09777211400638387, 0.4402524937396013],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False],
       fill_value=1e+20), 'param_kernel': masked_array(data=['rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf',
                         'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf',
                         'rbf', 'rbf', 'rbf', 'rbf'],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False],
       fill_value=np.str_('?'),
             dtype=object), 'params': [{'C': np.float64(37.55401188473625), 'gamma': np.float64(0.9508143064099162), 'kernel': 'rbf'}, {'C': np.float64(73.2993941811405), 'gamma': np.float64(0.5987584841970366), 'kernel': 'rbf'}, {'C': np.float64(15.701864044243651), 'gamma': np.float64(0.15609452033620264), 'kernel': 'rbf'}, {'C': np.float64(5.908361216819946), 'gamma': np.float64(0.8662761457749352), 'kernel': 'rbf'}, {'C': np.float64(60.21150117432088), 'gamma': np.float64(0.7081725777960455), 'kernel': 'rbf'}, {'C': np.float64(2.1584494295802448), 'gamma': np.float64(0.9700098521619943), 'kernel': 'rbf'}, {'C': np.float64(83.34426408004217), 'gamma': np.float64(0.21243911067827614), 'kernel': 'rbf'}, {'C': np.float64(18.282496720710064), 'gamma': np.float64(0.1835045098534338), 'kernel': 'rbf'}, {'C': np.float64(30.524224295953772), 'gamma': np.float64(0.5248564316322378), 'kernel': 'rbf'}, {'C': np.float64(43.294501864211576), 'gamma': np.float64(0.2913291401980419), 'kernel': 'rbf'}, {'C': np.float64(61.28528947223795), 'gamma': np.float64(0.13959386065204182), 'kernel': 'rbf'}, {'C': np.float64(29.314464853521816), 'gamma': 
np.float64(0.3664618432936917), 'kernel': 'rbf'}, {'C': np.float64(45.706998421703595), 'gamma': np.float64(0.7852759613930136), 'kernel': 'rbf'}, {'C': np.float64(20.067378215835976), 'gamma': np.float64(0.5143344384136116), 'kernel': 'rbf'}, {'C': np.float64(59.34145688620425), 'gamma': np.float64(0.04655041271999773), 'kernel': 'rbf'}, {'C': np.float64(60.85448519014384), 'gamma': np.float64(0.17062412368729152), 'kernel': 'rbf'}, {'C': np.float64(6.605159298527951), 'gamma': np.float64(0.9489855372533332), 'kernel': 'rbf'}, {'C': np.float64(96.66320330745593), 'gamma': np.float64(0.8084973481164611), 'kernel': 'rbf'}, {'C': np.float64(30.56137691733707), 'gamma': np.float64(0.09777211400638387), 'kernel': 'rbf'}, {'C': np.float64(68.52330265121569), 'gamma': np.float64(0.4402524937396013), 'kernel': 'rbf'}], 'split0_test_score': array([0.5   , 0.4875, 0.8125, 0.4875, 0.4875, 0.5   , 0.8375, 0.8375,
       0.4875, 0.7125, 0.8375, 0.575 , 0.4875, 0.4875, 0.875 , 0.825 ,
       0.5   , 0.4875, 0.8375, 0.525 ]), 'split1_test_score': array([0.5125, 0.5125, 0.875 , 0.5125, 0.5125, 0.5125, 0.85  , 0.875 ,
       0.5   , 0.8   , 0.875 , 0.6875, 0.5125, 0.5   , 0.85  , 0.875 ,
       0.5125, 0.5125, 0.8375, 0.5625]), 'split2_test_score': array([0.5125, 0.5125, 0.875 , 0.5125, 0.5125, 0.5125, 0.8375, 0.85  ,
       0.5125, 0.725 , 0.875 , 0.6   , 0.5125, 0.5125, 0.8625, 0.85  ,
       0.5125, 0.5125, 0.875 , 0.5375]), 'split3_test_score': array([0.5125, 0.5125, 0.8875, 0.5125, 0.5125, 0.5125, 0.8875, 0.9   ,
       0.5375, 0.7875, 0.8875, 0.65  , 0.5125, 0.55  , 0.85  , 0.9   ,
       0.5125, 0.5125, 0.875 , 0.5875]), 'split4_test_score': array([0.5125, 0.5125, 0.8   , 0.5125, 0.5125, 0.5125, 0.7625, 0.775 ,
       0.5125, 0.7125, 0.8125, 0.575 , 0.5125, 0.5125, 0.85  , 0.8   ,
       0.5125, 0.5125, 0.825 , 0.525 ]), 'mean_test_score': array([0.51  , 0.5075, 0.85  , 0.5075, 0.5075, 0.51  , 0.835 , 0.8475,
       0.51  , 0.7475, 0.8575, 0.6175, 0.5075, 0.5125, 0.8575, 0.85  ,
       0.51  , 0.5075, 0.85  , 0.5475]), 'std_test_score': array([0.005     , 0.01      , 0.03622844, 0.01      , 0.01      ,
       0.005     , 0.04062019, 0.04213075, 0.01658312, 0.03824265,
       0.02806243, 0.04444097, 0.01      , 0.0209165 , 0.01      ,
       0.03535534, 0.005     , 0.01      , 0.0209165 , 0.0242384 ]), 'rank_test_score': array([12, 16,  3, 16, 16, 12,  7,  6, 12,  8,  2,  9, 16, 11,  1,  3, 12,
       16,  3, 10], dtype=int32)}
"""

"""
Q1. How does RandomizedSearchCV find the optimal hyperparameters? What is the advantage compared to GridSearchCV?

    Method: RandomizedSearchCV generates combinations by **randomly sampling** from the hyperparameter **distributions** defined by the user, for `n_iter` times. It then performs cross-validation on each generated combination to find the optimal set.

    Advantage: When the search space is very large or there are many hyperparameters, it can significantly **reduce the long training time** of GridSearchCV. Unlike GridSearchCV, which tests every combination (exhaustive search), RandomizedSearchCV can allocate more time to exploring the values of more important hyperparameters, making the search **more efficient** (sampling search).

Q2. What is the role of the `n_iter` parameter in RandomizedSearchCV? What is the effect of increasing or decreasing this value?

    `n_iter` specifies the **number of hyperparameter combinations** that RandomizedSearchCV will randomly sample and explore.

    **Increasing `n_iter`**: Increases the number of combinations explored, thereby **raising the probability of finding the optimal hyperparameters**, but the training time also increases.

    **Decreasing `n_iter`**: Reduces the training time, but **increases the chance of failing to find the optimal hyperparameters**.

Q3. Why are distributions, like `uniform(loc, scale)`, defined in the code? Is it possible to use a list of values, like in GridSearchCV?

    RandomizedSearchCV uses **distributions** to allow for **random sampling of values from a continuous range**.
    This helps to cover a broader search space more efficiently than GridSearchCV, which only explores pre-defined, discrete values.

    Yes, it is possible to use a **list of values**. For example, by passing a list like `{'C': [0.1, 1, 10]}`, RandomizedSearchCV will randomly select values only from within that list.

Q4. Is there a possibility that RandomizedSearchCV fails to find the optimal hyperparameters? If so, what are the ways to mitigate this drawback?

    Yes, because RandomizedSearchCV uses a **random search** method, it does not explore every combination like GridSearchCV. Therefore, there is a chance that it might miss the truly optimal combination.

    Mitigation Methods:
        **Increase the search iterations (`n_iter`)**: A sufficient number of iterations increases the probability of finding a good combination.
        **Two-stage Tuning**: Use RandomizedSearchCV to find the approximate range of the best hyperparameters first, and then use GridSearchCV to perform a **more refined search** around that specific range. This balances efficiency and accuracy.
"""
