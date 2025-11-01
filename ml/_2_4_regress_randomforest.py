"""
Supervised Learning:
    Classification: Used to categorize data into pre-defined classes or categories.
        Examples: sklearn.svm.SVC, sklearn.ensemble.RandomForestClassifier
    Regression: Used to predict a continuous numerical value.
        Examples: sklearn.linear_model.LinearRegression, sklearn.ensemble.RandomForestRegressor

RandomForestRegressor is an **Ensemble** model that builds multiple Decision Trees and
aggregates their predictions to produce the final result.
It is very similar to RandomForestClassifier, which is used for classification problems, but it is specialized for regression problems, which predict continuous values.

Core Principles of RandomForestRegressor:
1.  **Bagging (Bootstrap Aggregating)**:
    Multiple subsets (Bootstrap Samples) are created by randomly sampling the training data with replacement.
    Each subset is used to train an independent decision tree.

2.  **Random Feature Selection**:
    When splitting a node in each tree, only a subset of the total features is randomly selected to find the best split.
    This ensures that each tree is trained based on different features, reducing the correlation between trees and improving the model's generalization performance.

3.  **Prediction (Averaging)**:
    When making a prediction on new data, every decision tree in the forest outputs its individual predicted value.
    RandomForestRegressor uses the **average** of all these predicted values as the final prediction result.
    (Note: RandomForestClassifier uses Majority Voting.)

Through this process, it effectively solves the Overfitting problem that a single decision tree might have, leading to more stable and higher performance.
This demo code will walk through the learning process of RandomForestRegressor step-by-step.
"""

# 1. Import necessary libraries
from sklearn.datasets import make_regression  # Generate virtual data for regression
from sklearn.ensemble import RandomForestRegressor  # Random Forest Regression model
from sklearn.metrics import mean_squared_error, r2_score  # Model evaluation metrics
from sklearn.model_selection import train_test_split  # Data splitting

# 2. Generate example data
# Simulate a business scenario: predicting 'Used Car Price'.
# - n_samples=200: Data for 200 used car listings
# - n_features=10: 10 features (e.g., Year, Mileage, Engine Size, Brand, Accident History, etc.)
# - y (target variable): 'Used Car Price'
# - noise=20: Unpredictable factors influencing price (e.g., seller's urgency, negotiation skill, etc.)
# - random_state=42: Seed value for reproducible results
X, y = make_regression(n_samples=200, n_features=10, noise=20, random_state=42)

# Check the range of the generated data
# make_regression samples values primarily from a standard normal distribution (mean 0, std dev 1),
# so the value range is not restricted between -1 and 1.
print(f"Minimum value of generated feature (X) data: {X.min():.2f}")
print(f"Maximum value of generated feature (X) data: {X.max():.2f}")
print(f"Standard deviation of generated feature (X) data: {X.std():.2f}")
print(f"Mean of generated feature (X) data: {X.mean():.2f}")
print("-" * 50)

# 3. Data Splitting
# Split the entire data into training (train) and test sets.
#
# [Consideration on Training/Test Data Split Ratio]
# Various ratios like 80:20, 70:30, 75:25 are used, and there is no single correct answer.
# The choice mainly depends on the size of the overall dataset.
# - If the dataset is large (hundreds of thousands or more): A small test set ratio, such as 90:10 or 99:1, can be used,
#   as the number of test samples will still be sufficient for stable evaluation.
# - If the dataset is small (hundreds to tens of thousands): A ratio of 80:20 or 70:30 is generally used.
#   This is a balance point to ensure enough training data while maintaining evaluation reliability.
#
# In this example, we use the 80:20 ratio, which is one of the most widely used standard ratios.
# For more reliable validation, using Cross-Validation is recommended.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create Estimator Instance
# Create the RandomForestRegressor model.
#
# [Consideration on selecting the n_estimators value]
# n_estimators, the number of decision trees composing the forest, significantly affects model performance.
#
# - Is more always better?
#   Generally, a higher number of trees stabilizes the model's performance and reduces the variance of predictions.
#   However, beyond a certain level, performance improvement becomes negligible, and the phenomenon of **Diminishing Returns** occurs, where only training time and memory usage increase.
#
# - What is the Best Practice?
#   There is no absolute rule like "number of trees vs. number of data points" as the optimal value varies depending on the dataset size and complexity.
#   Instead, the following **Heuristic** approaches are widely used:
#   1. Starting Point: Starting with a value between 100 and 300 is generally a good initial point. (scikit-learn's default is 100.)
#   2. Check the Performance Curve: Visualize the change in Cross-Validation score or OOB (Out-of-Bag) score as the number of trees increases.
#      The most ideal method is to find the **Elbow point**, where performance stabilizes without significant further improvement.
#   3. Consider Computational Cost: There is no need to use significantly more trees beyond the point where performance improvement is minimal.
#      It's important to balance performance with computational cost.
#
# Conclusion: 100 is a reasonable default, but for optimal performance, it is best to find the optimal n_estimators value for your specific dataset using methods like Cross-Validation or GridSearchCV.
model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)

# 5. Model Training (fit)
# Use the model.fit(X, y) method to fit the model to the training data.
# Through this process, the model creates and trains multiple decision trees.
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

"""
Execution Example

Minimum value of generated feature (X) data: -2.64
Maximum value of generated feature (X) data: 2.93
--------------------------------------------------
Starting model training...
Model training complete!

Mean Squared Error (MSE): 9167.42
R-squared (R2 Score): 0.74
"""

"""
Review

Q1. How are RandomForestRegressor and RandomForestClassifier similar, and how are they different?

    - Similarities: Both use the **Ensemble Learning** technique, where they build multiple **Decision Trees** and aggregate their predictions.
      They also share main hyperparameters like n_estimators.

    - Differences: RandomForestClassifier aggregates tree predictions using **Majority Voting** to determine the final class label for classification,
      whereas RandomForestRegressor determines the final continuous value by taking the simple **Average** of the trees' predictions for regression.

Q2. How is the final predicted value determined in a regression problem using RandomForestRegressor?

    - RandomForestRegressor collects the predicted values from each decision tree and uses the **simple average** of those values as the final predicted value.
      For example, if 100 trees each output a prediction, the average of those 100 values becomes the final prediction result.

Q3. Besides the n_estimators parameter, what other important parameters can influence the performance of RandomForestRegressor?

    - **max_depth**: Limits the maximum depth of each decision tree.
      Setting this value too deep can lead to **Overfitting**.
      The default is None, allowing trees to grow fully until all leaves are pure or contain less than min_samples_split samples.

    - **min_samples_split**: Specifies the minimum number of samples required to split an internal node.
      Setting this value higher simplifies the model and can prevent overfitting.

    - **max_features**: Specifies the maximum number of features (predictors) that each decision tree considers when looking for the best split.
      Adjusting this value can increase tree diversity.

Q4. Between mean_squared_error and r2_score, which metric is more helpful for intuitively understanding model performance? Why?

    - **r2_score** is generally more helpful for intuitively understanding model performance.

    - **r2_score** is expressed as a value between 0 and 1, where a value closer to 1 means the model explains the data well.
      This can be interpreted as 'how much percentage of the data's variability the model explains,' making it easy to understand.

    - In contrast, **mean_squared_error** is the average of squared errors, lacking an inherent unit, making it difficult to directly compare across different datasets.
      While a lower value is better, it's hard to judge whether an MSE value of 20 is good or bad without a benchmark.
"""
