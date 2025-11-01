"""
Feature Preprocessing and Engineering

Various techniques to process data into a form suitable for model training.
    sklearn.preprocessing.OneHotEncoder: Converts categorical data into a format machine learning models can understand using one-hot encoding. (e.g., 'Seoul', 'Busan' -> [1, 0], [0, 1])
    sklearn.preprocessing.MinMaxScaler: Scales data to values between 0 and 1. This is another important scaling method besides StandardScaler.
    sklearn.feature_extraction.text: Functionality for extracting features from text data. CountVectorizer and TfidfVectorizer are representative examples.

OneHotEncoder:
    OneHotEncoder is a preprocessing tool that converts categorical data into numerical data so that machine learning models can learn from it.
    This encoding scheme converts each category into independent binary (0 or 1) features,
    which is useful for preventing the model from misinterpreting sequence or relationships between categories.
"""

# 1. Import necessary libraries
import pandas as pd
from sklearn.compose import ColumnTransformer  # Used to apply different transformers to multiple features
from sklearn.preprocessing import OneHotEncoder

# 2. Generate example data
# Create a DataFrame with categorical features named 'City' and 'Weather'.
# 'City' has 3 unique categories, and 'Weather' has 2 unique categories.
data = {
    "City": ["Seoul", "Busan", "Seoul", "Jeju", "Busan"],
    "Weather": ["Sunny", "Rainy", "Sunny", "Sunny", "Rainy"],
    "Temperature": [25, 20, 26, 22, 21],
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)
print("-" * 50)

# 3. Create OneHotEncoder instance and apply
# drop='first': Excludes the first category to avoid the multicollinearity problem.
#                (e.g., if 'City_Jeju' and 'City_Seoul' are both 0, it implies 'City_Busan')
# handle_unknown='ignore': Ignores new categories that appear in the test data but were not in the training data.
# sparse_output=False: Outputs a regular array (ndarray) instead of a sparse matrix.
encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)

# Use fit_transform() to encode the 'City' column.
# The encoder learns the order of 'Busan', 'Jeju', 'Seoul' (alphabetical order).
city_encoded = encoder.fit_transform(df[["City"]])

# Convert the encoded result back to a DataFrame for checking
city_encoded_df = pd.DataFrame(city_encoded, columns=encoder.get_feature_names_out(["City"]))
print(" 'City' column after OneHotEncoder application:")
print(city_encoded_df)
print("-" * 50)

# 4. Simultaneous Processing of Multiple Columns using ColumnTransformer
# In real data, different preprocessing steps must be applied to different columns.
# ColumnTransformer performs this process efficiently.
# (1) List of columns to apply OneHotEncoder to ('City', 'Weather')
categorical_features = ["City", "Weather"]

# (2) Create ColumnTransformer instance
# transformers: A list of tuples (transformer name, transformer instance, list of columns to apply to)
preprocessor = ColumnTransformer(
    transformers=[
        (
            "onehot",
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            categorical_features,
        )
    ],
    remainder="passthrough",  # Columns not specified in 'transformers', like 'Temperature', are kept as is.
)

# Use fit_transform() to preprocess multiple columns at once.
X_processed = preprocessor.fit_transform(df)

# Check the final result
processed_df = pd.DataFrame(
    X_processed,
    # Create a new list of column names by combining encoded and original column names
    columns=preprocessor.get_feature_names_out(),
)
print("Full DataFrame preprocessing result using ColumnTransformer:")
print(processed_df)

"""
Execution Example:

Original DataFrame:
    City Weather  Temperature
0  Seoul   Sunny           25
1  Busan   Rainy           20
2  Seoul   Sunny           26
3   Jeju   Sunny           22
4  Busan   Rainy           21
--------------------------------------------------
 'City' column after OneHotEncoder application:
    City_Jeju  City_Seoul
0         0.0         1.0
1         0.0         0.0        // City_Busan is removed by drop='first'
2         0.0         1.0
3         1.0         0.0
4         0.0         0.0        // City_Busan is represented when both columns are 0
--------------------------------------------------
Full DataFrame preprocessing result using ColumnTransformer:
   onehot__City_Jeju  onehot__City_Seoul  onehot__Weather_Sunny  remainder__Temperature
0                0.0                 1.0                    1.0                    25.0
1                0.0                 0.0                    0.0                    20.0
2                0.0                 1.0                    1.0                    26.0
3                1.0                 0.0                    1.0                    22.0
4                0.0                 0.0                    0.0                    21.0
"""

"""
Code Review Questions

Q1. What is the main purpose of using OneHotEncoder, and what problems can arise if categorical data is used as is?

    - The purpose of OneHotEncoder is to convert unordered categorical data, such as 'Seoul' or 'Busan', into numerical data that a machine learning model can recognize.

    - If categorical data is simply converted into numbers like [1, 2, 3], the model might mistakenly assume a sequential order or magnitude relationship exists between these numbers. 
      For example, it might incorrectly learn that '3' is twice as important as '1', or that '2' is the midpoint of '1' and '3'. OneHotEncoder prevents this misinterpretation.

Q2. What role does the drop='first' parameter in OneHotEncoder play, and what benefit does its use provide?

    - drop='first' removes the column corresponding to the first category within each categorical feature.

    - This option helps avoid the problem of **multicollinearity**. For example, if the City_Jeju and City_Seoul columns are both 0, it implies the city is Busan. 
      Since one category can be represented by the combination of the others, removing this redundant information increases the **stability** of the model.

      Not removing multicollinearity can lead to issues such as:
      
        **Instability of Regression Coefficients (Weights):**
            When multicollinearity is present, it's difficult for regression models to accurately measure the importance of each variable.
            Because the variables are highly correlated, a small change in one variable's value can cause the model's calculated regression coefficient (weight) to fluctuate greatly.
            This reduces the model's interpretability and prediction reliability.

        **Incorrect Coefficient Signs:**
            Coefficients with the opposite sign to their actual effect might appear. For example, while it's generally accepted that higher income leads to higher spending, multicollinearity might cause the income variable's coefficient to be calculated as negative.

        **Difficulty in Determining Statistical Significance:**
            The p-value of each independent variable may appear high, leading to the incorrect conclusion that a variable is not statistically significant, even if it is actually important.

Q3. What benefits are gained by using ColumnTransformer, and what does the remainder='passthrough' parameter signify?

    - ColumnTransformer allows the integration and management of different preprocessing steps (e.g., a scaler for some columns, an encoder for others) that need to be applied to different columns, all within a single object. 
      This enhances code readability and is very useful for automating the preprocessing-modeling process when combined with a Pipeline.

    - **remainder='passthrough'** means that all remaining columns not explicitly specified in the `transformers` list should be kept as is, without any transformation. In the demo code, the 'Temperature' column falls into this category.

Q4. In the demo code's OneHotEncoder result, where did 'Busan' from the 'City' column go, and why are only 'City_Seoul' and 'City_Jeju' left?

    - The city of **'Busan'** is implicitly represented by the remaining columns because of the **drop='first'** parameter.
    
    - OneHotEncoder sorts the unique categories alphabetically: **'Busan'**, 'Jeju', 'Seoul'.
    - With `drop='first'`, the column corresponding to the first category, **'City\_Busan'**, is removed to prevent multicollinearity.
    - Therefore, only the columns **'City\_Jeju'** and **'City\_Seoul'** remain. 'Busan' is signified when both remaining columns have a value of **0**.
"""
