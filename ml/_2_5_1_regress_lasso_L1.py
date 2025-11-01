"""
In addition to the models covered so far, scikit-learn includes a variety of other algorithms.
    Ensemble Models: There are models that offer stronger performance than RandomForest, such as GradientBoostingClassifier, XGBoost, and LightGBM. (LightGBM and XGBoost require separate installation)
    Kernel-Based Models: Besides SVC, there are kernel-based models like SVR (for regression).
    Linear Models: In addition to LinearRegression, there are models like Lasso and Ridge, which prevent overfitting through Regularization.

Lasso (Least Absolute Shrinkage and Selection Operator):
    Lasso is a linear regression model with L1 regularization applied.
    It is used to reduce model complexity and prevent Overfitting by adding a penalty term to the regular linear regression model.
    Notably, L1 regularization has the characteristic of forcing the coefficients of unnecessary features to zero,
    which provides a secondary benefit of **Feature Selection**.
"""

# 1. Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 2. Generate example data
# Simulate a business scenario: predicting 'House Price'.
# Create a dataset with 100 house records (n_samples) and 5 features (n_features).
#
# - y (Target variable): 'House Price'
# - Significant features (affecting price):
#   - X[:, 0]: 'House Size (m²)' (Largest effect, coefficient=3)
#   - X[:, 1]: 'Number of Rooms' (Medium effect, coefficient=2)
#   - X[:, 2]: 'Distance to Subway Station' (Smallest effect, coefficient=1)
# - Unnecessary features (no correlation with price):
#   - X[:, 3]: 'Local Park Visitor Count'
#   - X[:, 4]: 'Daily Average Sales of Nearby Shops'
#
# The goal of this demo is to confirm that the Lasso model forces the coefficients of these unnecessary features (x4, x5) to zero,
# automatically selecting only the important features.
np.random.seed(42)
X = np.random.randn(100, 5)
y = 3 * X[:, 0] + 2 * X[:, 1] + 1 * X[:, 2] + np.random.randn(100) * 2

# Create a dummy dataset that roughly matches the comments.
# Significant features (affecting price):
#   - X[:, 0]: 'House Size (m²)' (Largest effect, coefficient=3)
#   - X[:, 1]: 'Number of Rooms' (Medium effect, coefficient=2)
#   - X[:, 2]: 'Distance to Subway Station' (Smallest effect, coefficient=1)
# X = np.random.rand(100, 5) * np.array([100, 10, 1, 1000, 100])  # Assign scale differences
# y = 3 * X[:, 0] + 2 * X[:, 1] + 1 * X[:, 2] + np.random.randn(100) * 5  # Add noise

# Check statistical information of the generated data
# np.random.randn() generates data following a standard normal distribution (mean 0, std dev 1).
# The mean and standard deviation of the actual generated data are very close to 0 and 1.
print("\n--- Statistical Information of Generated Data (X) ---")
print(f"Mean of all features (X): {X.mean():.4f}")
print(f"Standard deviation of all features (X): {X.std():.4f}")
print("-" * 50)

# 3. Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==============================================================================
# Section 1: Standard Linear Regression (Lasso Not Applied)
# Check the coefficients of the basic linear regression model without regularization.
# ==============================================================================
print("=== [Section 1] Standard Linear Regression Model ===")
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred_lin = lin_reg.predict(X_test)
mse_lin = mean_squared_error(y_test, y_pred_lin)

print(f"Coefficients of the Standard Linear Regression Model: {lin_reg.coef_}")
print(f"MSE (Error): {mse_lin:.4f}")
print("-" * 50)


# ==============================================================================
# Section 2: Lasso Model Application (L1 Regularization)
# Check the regularization effect and feature selection effect of the Lasso model.
# ==============================================================================
print("=== [Section 2] Lasso Model ===")

# Create a Lasso model instance
# alpha: The hyperparameter that controls the regularization strength.
#        A larger alpha increases regularization, pushing coefficients closer to 0.
lasso_reg = Lasso(alpha=1.0, random_state=42)
lasso_reg.fit(X_train, y_train)

y_pred_lasso = lasso_reg.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

print(f"Coefficients of the Lasso Model (alpha=1.0): {lasso_reg.coef_}")
print(f"MSE (Error): {mse_lasso:.4f}")

# Note that the coefficients of the unnecessary features (4th and 5th features) have become close to 0 or exactly 0.
print("\n- The coefficients of the unnecessary features converge to 0, demonstrating the feature selection effect.")
print("-" * 50)


# ==============================================================================
# Section 3: Visualization of Coefficient Change based on alpha value
# Visually confirm how the coefficients change according to the alpha value.
# ==============================================================================
print("=== [Section 3] Coefficient Change based on alpha value ===")
# np.logspace(-2, 2, 100) generates 100 values evenly spaced on a log scale
# from 10^-2 (0.01) to 10^2 (100).
# This allows effective observation of changes when alpha is both small and large.
# This code generates a total of 100 alpha values.
alphas = np.logspace(-2, 2, 100)
coefs = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)

plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale("log")
plt.xlabel("Alpha (Regularization Strength)")
plt.ylabel("Coefficients")
plt.title("Lasso Coefficients as a function of Alpha")
plt.legend([f"Feature {i + 1}" for i in range(5)])
plt.axis("tight")
plt.grid(True)
plt.show()

"""
Execution Example

--- Statistical Information of Generated Data (X) ---
Mean of all features (X): 0.0364
Standard deviation of all features (X): 0.9945
--------------------------------------------------
=== [Section 1] Standard Linear Regression Model ===
Coefficients of the Standard Linear Regression Model: [3.1631208  2.13159999 1.06827716 0.08499474 0.00957412]
MSE (Error): 3.2356
--------------------------------------------------
=== [Section 2] Lasso Model ===
Coefficients of the Lasso Model (alpha=1.0): [ 1.69950905  1.03821904  0.0140734  -0.          -0.        ]
MSE (Error): 7.3102

- The coefficients of the unnecessary features converge to 0, demonstrating the feature selection effect.
--------------------------------------------------
=== [Section 3] Coefficient Change based on alpha value ===
"""

"""
Review

Q1. What is the role of Regularization in the Lasso model, and why does it help prevent Overfitting?

    - Regularization acts to **penalize the complexity of the model** to prevent it from excessively fitting the data.
      Lasso specifically imposes a penalty proportional to the sum of the absolute values of the model's coefficients (L1 penalty).

    - Overfitted models often have very large coefficient values for specific features.
      Regularization helps prevent overfitting by suppressing these coefficients, making the model simpler and more generalized.

Q2. What is the biggest difference between Lasso and Standard Linear Regression, and explain why Lasso is considered useful for 'Feature Selection'.

    - Biggest Difference: Lasso adds an **L1 regularization term** to the loss function, whereas Standard Linear Regression has no regularization term.

    - Feature Selection: Lasso's L1 regularization has the property of forcing coefficient values to exactly **zero** if the penalty is too large.
      This effectively causes the model to automatically exclude unnecessary or unimportant features, which is why Lasso is valued for feature selection.

Q3. What is the role of the Lasso model's hyperparameter, alpha, and how does the model change as alpha increases?

    - **alpha** is the hyperparameter that controls the **strength of regularization** in the Lasso model.

    - As the alpha value **increases**, the regularization becomes stronger.
      The model's penalty increases, causing the coefficient values to converge to 0 more rapidly.
      This means the model becomes simpler and the effect of preventing overfitting becomes stronger.

    - As the alpha value **decreases**, the regularization becomes weaker.
      When alpha approaches 0, Lasso becomes nearly identical to the Standard Linear Regression model.

Q4. In the demo code's visualization result, we can see the coefficients approaching 0 as the alpha value increases. What does this signify?

    - This visualization shows how the model's complexity decreases as the regularization strength (alpha) increases.

    - When alpha is small, all feature coefficients have significant values. However, as alpha gradually increases, the coefficients of less important features (e.g., the 4th and 5th features in the demo) are forced to 0 first.
      Eventually, even the coefficients of more important features are pushed closer to 0.
      This visually demonstrates Lasso's process of progressively simplifying the model and eliminating unnecessary features.

Q5. So, what is the model tuning method?

    - A common tuning method is to adjust the alpha value to find the optimal value that **minimizes the MSE**.

    - Finding the optimal alpha value that **maximizes the r2_score** is also a good method.
      Since $r^2 = 1 - (\text{MSE} / \text{Var}(y))$, maximizing $r^2$ is equivalent to minimizing MSE.

    It is typical to use techniques like **GridSearchCV** to test various alpha values and find the optimal one.
"""
