"""
Supervised Learning:
    Classification: Used to classify data into pre-defined classes or categories.
        Examples: sklearn.svm.SVC, sklearn.ensemble.RandomForestClassifier
    Regression: Used to predict a continuous numerical value.
        Examples: sklearn.linear_model.LinearRegression, sklearn.ensemble.RandomForestRegressor

LR:
    LinearRegression is the most fundamental linear regression model,
    used to predict continuous values by modeling the linear relationship between data points.
    This demo code will walk through the learning process of LinearRegression step-by-step.
"""

# 1. Import necessary libraries
from sklearn.datasets import make_regression  # Generate virtual data for regression
from sklearn.linear_model import LinearRegression  # Linear Regression model
from sklearn.metrics import mean_squared_error, r2_score  # Model evaluation metrics
from sklearn.model_selection import train_test_split  # Data splitting

# 2. Generate example data
# Simulate a business scenario: predicting 'Daily Revenue' based on 'Daily Advertising Spend'.
# - n_samples=200: Data for 200 days
# - n_features=1: 1 feature (independent variable), which is 'Daily Advertising Spend'
# - y (target variable): 'Daily Revenue'
# - noise=20: Unpredictable factors influencing revenue besides ad spend (weather, competitor events, etc.)
# - random_state=42: Seed value for reproducible results
X, y = make_regression(n_samples=200, n_features=1, noise=20, random_state=42)

# 3. Data Splitting
# Split the entire data into training (train) and test sets.
# The training set (80%) is used for model training, and the test set (20%) is used for model performance evaluation.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create Estimator Instance
# Create the LinearRegression model.
model = LinearRegression()

# 5. Model Training (fit)
# Use the model.fit(X, y) method to fit the model to the training data.
# Through this process, the model finds the coefficient (slope) and intercept of the regression line that best fits the data.
print("Starting model training...")
model.fit(X_train, y_train)
print("Model training complete!")

# 6. Prediction (predict)
# Use the model.predict(X) method to perform predictions on the test data using the trained model.
# The prediction results will be continuous numerical values (regression values).
y_pred = model.predict(X_test)

# 7. Model Evaluation
# Evaluate the model's performance by comparing the actual values (y_test) with the predicted values (y_pred).

# Mean Squared Error (MSE): The average of the squared differences (errors) between predicted and actual values.
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error (MSE): {mse:.2f}")

# R-squared (R2 Score): A metric indicating how well the model explains the variance. Closer to 1 is better.
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R2 Score): {r2:.2f}")

# Output the model's coefficient and intercept
print("\nModel Coefficient (slope):", model.coef_[0])
print("Model Intercept:", model.intercept_)

"""
Starting model training...
Model training complete!

Mean Squared Error (MSE): 437.55
R-squared (R2 Score): 0.94

Model Coefficient (slope): 86.5115419768739
Model Intercept: 2.4461021846792352
"""

"""
Review

Q1. What type of machine learning problem is the LinearRegression model primarily used for? What is the difference between it and SVC or RandomForestClassifier?

    - LinearRegression is a Supervised Learning model used for **Regression** problems, meaning predicting a continuous numerical value.

    - SVC or RandomForestClassifier are used for **Classification** problems, meaning dividing data into predefined categories.
      The biggest difference is that Regression predicts a continuous value, while Classification predicts discrete class labels.

Q2. When the fit() method is called on a regression model, what are the main parameters the model learns?

    - Through the fit() method, the LinearRegression model learns the **coefficient (slope)** and **intercept** that best represent the relationship between the input data (X_train) and the target (y_train).
      These parameters are the core elements that define the model.

Q3. What aspects of the model are assessed by mean_squared_error and r2_score, respectively?

    - **mean_squared_error (MSE)**: The average of the squared differences (errors) between predicted and actual values, used to measure prediction accuracy. A value closer to 0 means the model's predictions are more accurate.

    - **r2_score (R-squared)**: An indicator of how well the model explains the data's variance. It ranges between 0 and 1, and a value closer to 1 suggests the model explains the data well.

Q4. How does the noise parameter affect the training and performance evaluation of the regression model?

    - The **noise** parameter represents the randomness or scatter added to the data.

    - A larger value means the data points are scattered further from the regression line, making it harder for the model to learn the underlying pattern.
      This leads to a decrease in model performance.

    - Therefore, a larger noise value typically results in an increase in the **mean_squared_error** and a decrease in the **r2_score**.
"""
