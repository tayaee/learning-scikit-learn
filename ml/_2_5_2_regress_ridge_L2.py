"""
In addition to the models covered so far, scikit-learn includes a variety of other algorithms.
    Ensemble Models: There are models that offer stronger performance than RandomForest, such as GradientBoostingClassifier, XGBoost, and LightGBM. (LightGBM and XGBoost require separate installation)
    Kernel-Based Models: Besides SVC, there are kernel-based models like SVR (for regression).
    Linear Models: In addition to LinearRegression, there are models like Lasso and Ridge, which prevent overfitting through Regularization.

Ridge is a linear regression model with **L2 regularization** applied.
    Similar to Lasso, it is used to reduce model complexity and prevent Overfitting by adding a penalty term to the standard linear regression.
    While Lasso's L1 regularization penalizes the sum of the absolute values of the coefficients,
    Ridge's L2 regularization penalizes the sum of the **squares** of the coefficients.
    Because of this, it drives the coefficients close to 0 but **does not** force them to become exactly 0.
"""

# 1. Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 2. Generate example data
# Create a dataset with 100 samples and 5 features.
# The data includes unnecessary features (x4, x5), identical to the Lasso example, to compare the behavior of the Ridge model.
np.random.seed(42)
X = np.random.randn(100, 5)
y = 3 * X[:, 0] + 2 * X[:, 1] + 1 * X[:, 2] + np.random.randn(100) * 2

# 3. Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==============================================================================
# Section 1: Standard Linear Regression (Ridge Not Applied)
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
# Section 2: Ridge Model Application (L2 Regularization)
# Check the regularization effect of the Ridge model.
# ==============================================================================
print("=== [Section 2] Ridge Model ===")

# Create a Ridge model instance
# alpha: The hyperparameter that controls the regularization strength.
#        A larger alpha increases regularization, pushing coefficients closer to 0.
ridge_reg = Ridge(alpha=1.0, random_state=42)
ridge_reg.fit(X_train, y_train)

y_pred_ridge = ridge_reg.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

print(f"Coefficients of the Ridge Model (alpha=1.0): {ridge_reg.coef_}")
print(f"MSE (Error): {mse_ridge:.4f}")

# Note that the coefficients are close to 0 but, unlike Lasso, they have not become exactly 0.
print("\n- The coefficients of all features have been shrunk toward 0 but have not become 0.")
print("-" * 50)


# ==============================================================================
# Section 3: Visualization of Coefficient Change based on alpha value
# Visually confirm how the coefficients change according to the alpha value.
# ==============================================================================
print("=== [Section 3] Coefficient Change based on alpha value ===")
alphas = np.logspace(-2, 2, 100)
coefs = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)

plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale("log")
plt.xlabel("Alpha (Regularization Strength)")
plt.ylabel("Coefficients")
plt.title("Ridge Coefficients as a function of Alpha")
plt.legend([f"Feature {i + 1}" for i in range(5)])
plt.axis("tight")
plt.grid(True)
plt.show()

"""
Execution Example

=== [Section 1] Standard Linear Regression Model ===
Coefficients of the Standard Linear Regression Model: [3.1631208  2.13159999 1.06827716 0.08499474 0.00957412]
MSE (Error): 3.2356
--------------------------------------------------
=== [Section 2] Ridge Model ===
Coefficients of the Ridge Model (alpha=1.0): [3.10017058 2.09502553 1.05381601 0.0829495  0.00408486]
MSE (Error): 3.2309

- The coefficients of all features have been shrunk toward 0 but have not become 0.
--------------------------------------------------
=== [Section 3] Coefficient Change based on alpha value ===
"""

"""
Review

Q1. What is the role of L2 regularization in the Ridge model, and how does it differ from Lasso's L1 regularization?

    - Ridge's L2 regularization imposes a penalty proportional to the **sum of the squares** of the model's coefficients.
      This penalty serves to prevent the coefficient values from becoming too large, thereby reducing model complexity and preventing overfitting.

    - Difference from Lasso: Lasso's L1 regularization penalizes the sum of the **absolute values** of the coefficients,
      which causes the coefficients of less important features to be forced to **exactly 0**.
      In contrast, Ridge's L2 regularization only shrinks the coefficient values close to 0 but does not make them exactly 0.

Q2. Why does Ridge, unlike Lasso, lack a 'Feature Selection' effect?

    - Because L2 regularization adds the sum of the squared coefficients to the loss function,
      it can make coefficient values small but generally cannot make them exactly 0.
      Since the coefficients do not become exactly 0,
      Ridge does not have the feature selection effect of excluding unnecessary features from the model, unlike Lasso.
      Instead, it shrinks all coefficients slightly to improve the model's stability.

Q3. Is the role of the hyperparameter alpha in the Ridge model the same as in Lasso? What happens to the model when alpha is very large?

    - Yes, the **alpha** in the Ridge model, like in Lasso, controls the **regularization strength**.
      A larger alpha results in stronger regularization.

    - If alpha is a very large value, the regularization penalty becomes overwhelmingly large, causing all the model's coefficient values to converge very close to 0.
      In this case, the model becomes very simple (underfitting), where all predictions essentially converge to the mean of the target variable.

Q4. In general, which model should be considered first, Ridge or Lasso? Does a combined model exist?

    - Order of Consideration: **Ridge** is often considered first.
      Because Ridge shrinks all features slightly, it tends to offer more stable performance when features have high correlation with each other.
      Lasso is useful when there are many unnecessary features that need to be clearly eliminated.

    - Combined Model: Yes, a model that combines Lasso's L1 regularization and Ridge's L2 regularization exists, called **ElasticNet**.
      When it's unclear whether to use Lasso or Ridge, using ElasticNet allows utilizing the benefits of both regularizations.
      ElasticNet takes the best of both worlds, reducing model complexity while also achieving a feature selection effect, which leads to stable performance in various data environments.

    - The minimization objective function used by Lasso's fit() function is:
        $$\text{Minimize} \quad ||y - X\beta||^2_2 + \alpha ||\beta||_1$$

    - The minimization objective function used by Ridge's fit() function is:
        $$\text{Minimize} \quad ||y - X\beta||^2_2 + \alpha ||\beta||^2_2$$

    - The minimization objective function used by ElasticNet's fit() function is:
        $$\text{Minimize} \quad ||y - X\beta||^2_2 + \alpha_1 ||\beta||_1 + \alpha_2 ||\beta||^2_2$$

Q5. Model Trial Sequence

    - Since it is often difficult to know in advance which model is better,
      **ElasticNet** is typically tried first using a technique like GridSearchCV to find the optimal alpha value.

      If the optimal alpha for the L1 ratio ($\alpha_1$) is close to 1 (and $\alpha_2$ is close to 0), it suggests that Lasso would have been a good choice.

      If the optimal alpha for the L1 ratio is close to 0 (and $\alpha_2$ is close to 1), it suggests that Ridge would have been a good choice.

      If the optimal alpha for the L1 ratio is between 0 and 1, it suggests that ElasticNet was the most appropriate choice.
"""
