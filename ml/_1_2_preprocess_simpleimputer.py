"""
Preprocessing:
    Transforms the data into a suitable format for the model to learn. This includes handling missing values, data scaling, and feature extraction.
    Example: sklearn.preprocessing.StandardScaler, sklearn.impute.SimpleImputer

SimpleImputer:
    SimpleImputer is a Transformer used to fill missing values in the data preprocessing stage.
    By filling missing values using a specific strategy (e.g., mean, median, most frequent, etc.),
    it makes the data containing missing values suitable for machine learning models to train on.
"""

# 1. Import necessary libraries
import numpy as np
import pandas as pd  # Used to handle data as a DataFrame
from sklearn.impute import SimpleImputer  # SimpleImputer transformer
from sklearn.model_selection import train_test_split  # Data splitting

# 2. Generate example data
# Create data containing missing values (np.nan).
# Assume 'Age' column is numerical and 'Embarked' column is categorical.
data = {
    "Age": [20, 25, 30, np.nan, 40, 45, 50, np.nan],
    "Fare": [100, 150, np.nan, 200, 250, 300, 350, 400],
    "Embarked": ["S", "C", "S", "Q", np.nan, "S", "C", "S"],
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# 3. Separate numerical and categorical data
# SimpleImputer is primarily used to process numerical data.
# Categorical data may require a different strategy, so they are separated.
numeric_data = df[["Age", "Fare"]]
categorical_data = df[["Embarked"]]

# 4. Apply SimpleImputer to numerical data
# Create an instance of the SimpleImputer estimator (transformer)
# strategy='mean': Fills missing values with the mean of the feature.
imputer_numeric = SimpleImputer(strategy="mean")

# fit_transform: Calculates the mean (fit) for the training data and fills missing values (transform).
numeric_data_imputed = imputer_numeric.fit_transform(numeric_data)

# 5. Apply SimpleImputer to categorical data
# strategy='most_frequent': Fills missing values with the most frequent value (mode) of the feature.
imputer_categorical = SimpleImputer(strategy="most_frequent")
categorical_data_imputed = imputer_categorical.fit_transform(categorical_data)

# 6. Check results
print("\nImputed Numerical Data:")
print(pd.DataFrame(numeric_data_imputed, columns=["Age", "Fare"]))
print("\nImputed Categorical Data:")
print(pd.DataFrame(categorical_data_imputed, columns=["Embarked"]))

# 7. (Bonus) Example of Imputer Application after Train/Test Split
# Apply Imputer after data splitting
X_train, X_test = train_test_split(numeric_data, test_size=0.3, random_state=42)

# Fit on training data (calculate mean)
imputer_split = SimpleImputer(strategy="mean")
imputer_split.fit(X_train)

# Transform the training data using fit_transform
X_train_imputed = imputer_split.transform(X_train)

# Transform the test data using transform only (uses the mean from training data)
X_test_imputed = imputer_split.transform(X_test)

print("\nTraining data statistics (mean):", imputer_split.statistics_)
print("\nImputed Training Data:")
print(pd.DataFrame(X_train_imputed, columns=["Age", "Fare"]))
print("\nImputed Test Data:")
print(pd.DataFrame(X_test_imputed, columns=["Age", "Fare"]))

"""
Original DataFrame:
    Age   Fare Embarked
0  20.0  100.0        S
1  25.0  150.0        C
2  30.0    NaN        S
3   NaN  200.0        Q
4  40.0  250.0      NaN
5  45.0  300.0        S
6  50.0  350.0        C
7   NaN  400.0        S

Imputed Numerical Data:
    Age   Fare
0  20.0  100.0
1  25.0  150.0
2  30.0  250.0
3  35.0  200.0
4  40.0  250.0
5  45.0  300.0
6  50.0  350.0
7  35.0  400.0

Imputed Categorical Data:
  Embarked
0        S
1        C
2        S
3        Q
4        S
5        S
6        C
7        S

Training data statistics (mean): [ 40. 300.]

Imputed Training Data:
    Age   Fare
0  40.0  400.0
1  30.0  300.0
2  40.0  250.0
3  40.0  200.0
4  50.0  350.0

Imputed Test Data:
    Age   Fare
0  25.0  150.0
1  45.0  300.0
2  20.0  100.0
"""

"""
Q1. What problem is SimpleImputer used to solve?

    - SimpleImputer is used to solve the problem of filling **missing values** present in a dataset. 
      Since most machine learning algorithms cannot handle data containing missing values, they must be replaced with appropriate values before model training.

Q2. What are the main options for the strategy parameter, and when is each suitable?

    - 'mean': Fills missing values with the mean of the feature. Suitable when the data follows a normal distribution or when outliers do not significantly affect the mean.
    - 'median': Fills missing values with the median. A more robust choice when there are many outliers that could distort the mean.
    - 'most_frequent': Fills missing values with the **mode (most frequently occurring value)** of the feature. Can be used for both numerical and categorical data.
    - 'constant': Fills missing values with a user-specified constant value.

Q3. In the code, strategy='mean' was used for numerical data, and strategy='most_frequent' for categorical data. Why are these different strategies applied?

    - **Numerical data** (Age, Fare) consists of continuous values, where calculating statistical indicators like the mean or median is meaningful.
    - **Categorical data** (Embarked) consists of discrete classes (strings) for which the mean cannot be calculated. Therefore, filling missing values with the mode—the most frequently occurring value among categories like 'S', 'C', 'Q'—is the natural approach.

Q4. When applying SimpleImputer to the training and test data split by train_test_split, how should the fit() and transform() methods be used? Why is this process important?

    - **Training Data:** Use **fit_transform()** (or fit() then transform()) to **calculate the statistics** of the training data and then **fill** the missing values.
    - **Test Data:** Use **transform()** only.
    - This process is crucial to prevent **Data Leakage**. Missing values in the training data must be filled using statistics (e.g., mean) calculated from the training data, and **missing values in the test data must also be filled using those same training data statistics**. If the statistics of the test data were calculated separately (fit), the model would gain pre-knowledge of information it shouldn't have (the test data's mean), leading to an overestimation of model performance.
    - When implementing the same result with pandas' fillna, the key precaution is: If deciding to fill missing values with the mean, the mean must be calculated **from the training data** and then applied to **both the training and test data**. If the mean is calculated from the entire dataset (train + test) and then fillna is applied to the whole, information about the test data is leaked, potentially overestimating the model's performance.
"""
