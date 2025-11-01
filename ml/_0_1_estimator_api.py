"""
Estimator API

All machine learning algorithms in scikit-learn are implemented through a consistent object called an 'Estimator'.
An Estimator refers to any object that learns from data, and it includes various types such as classifiers, regression models, and transformers.
The core methods of the Estimator API are as follows:
    fit(X, y): The method for fitting (training) the model to the data. X represents the input data (features), and y represents the target labels (correct answers).
    predict(X): The method for performing a prediction on new data X using the trained model.
    transform(X): The method used in the data preprocessing stage. It transforms the input data and outputs it.
    fit_transform(X, y): A method that performs fit and transform sequentially, often used for efficiency.
"""

# 1. Import necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression  # Linear Regression Model (Estimator)
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.preprocessing import StandardScaler  # Data Scaling (Transformer)

# 2. Generate example data
# X: Independent variable (House size), y: Dependent variable (House price)
# Use reshape(-1, 1) to convert a 1D array into a 2D array.
# -1 allows the dimension size to be automatically inferred based on the array's length.
# scikit-learn expects input data in the form of a 2D array.
X = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50]).reshape(-1, 1)
y = np.array([12, 18, 22, 28, 33, 38, 45, 52, 58])

# 3. Data splitting
# Split the data into training (train) and testing (test) sets.
# The training set is used for model training, and the test set is used for model performance evaluation.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---
# 4. Example use of the Estimator API (Transformer - StandardScaler)
# StandardScaler is a transformer that standardizes data to have a mean of 0 and a standard deviation of 1.
scaler = StandardScaler()

# fit_transform(): Calculates the mean and standard deviation of the training data (fit) and standardizes the data (transform).
# The information about the training data (mean, standard deviation) is subsequently used to transform the test data.
# It sets the mean to 0 and the standard deviation to 1. It does not assume a normal distribution.
X_train_scaled = scaler.fit_transform(X_train)

# transform(): The test data is only standardized using the statistics (mean, standard deviation) from the training data.
# Using fit on the test data can cause data leakage.
X_test_scaled = scaler.transform(X_test)

# ---
# 5. Example use of the Estimator API (Regression Model - LinearRegression)
# LinearRegression is an estimator that implements the linear regression model.
model = LinearRegression()

# fit(X, y): Trains the model using the training data (X_train_scaled, y_train).
# Through this process, the model finds the optimal regression coefficients.
model.fit(X_train_scaled, y_train)

# ---
# 6. Prediction (predict)
# predict(X): Performs a prediction on the test data using the trained model.
# For prediction, the data must be scaled identically to the data used for training.
predictions = model.predict(X_test_scaled)

# 7. Output results
print("Original Test Data:\n", X_test)
print("Scaled Test Data:\n", X_test_scaled)
print("Actual House Prices:\n", y_test)
print("Predicted House Prices:\n", predictions)

# Output the model's coefficient and intercept
print("\nModel's Coefficient:", model.coef_)
print("Model's Intercept:", model.intercept_)

"""
Execution Example:

Original Test Data:
 [[45]
 [15]
 [35]]
Scaled Test Data:
 [[ 1.21386547]
 [-1.08609016]
 [ 0.4472136 ]]
Actual House Prices:
 [52 18 38]
Predicted House Prices:
 [51.2244898  16.69387755 39.71428571]  

Model's Coefficient: [15.01359928]
Model's Intercept: 33.0
"""

"""
Code Review

Q1. What API role do LinearRegression and StandardScaler each perform?
    
    - **LinearRegression** is an **Estimator** that implements a supervised learning regression model.
      It is responsible for learning data patterns to predict continuous values.

    - **StandardScaler** acts as a **Transformer** that standardizes data features.
      It is used in the preprocessing stage to allow the model to learn the data more efficiently.

Q2. Why are the fit_transform() and transform() methods used on the StandardScaler instance, 'scaler', with X_train and X_test, respectively? Why shouldn't fit_transform() be used on X_test?
    
    - Reason for using fit_transform() on X_train:
      To **calculate (fit)** the mean and standard deviation for the training data X_train, and then **standardize (transform)** the data using those statistics.
      The model will be trained with these statistics.
    
    - Reason for using only transform() on X_test:
      The X_test data must be treated as unknown data that does not participate in training.
      If fit_transform() is used to calculate statistics for X_test, information about the training data can **leak (Data Leakage)** into the test data, making it impossible to accurately evaluate the model's generalization performance.
      Therefore, only standardization should be performed, applying the statistics obtained from the training data.

Q3. What kind of data should the predict() method accept as an argument? Why must X_test_scaled be used instead of the original X_test?
    
    - The predict() method must accept data that has been preprocessed in the exact same way as the data used for model training.
    
    - In this code, the LinearRegression model was trained using the StandardScaler-standardized data, X_train_scaled.
      Therefore, during prediction, the identically standardized X_test_scaled must be used for the model to make correct predictions.

Q4. At what stage does machine learning model training occur? And what role does the fit() method play at that time?

    - Machine learning model training occurs at stage 5 of the code when `model.fit(X_train_scaled, y_train)` is called.

    - At that moment, the fit() method is responsible for finding the optimal **weights (coefficients)** and **bias (intercept)** based on the given training data (X_train_scaled) and correct answers (y_train).
      Through this process, the model grasps the data's trends and becomes capable of performing predictions on new data.
"""
