"""
ColumnTransformer is a tool that allows different preprocessing transformers to be applied to different columns of a dataset.
Using it together with Pipeline enables the integration of complex preprocessing steps into a single workflow,
building a much more powerful and practical machine learning pipeline.
"""

# 1. Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer  # Apply different preprocessing to multiple columns
from sklearn.ensemble import RandomForestClassifier  # Final model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline  # Pipeline API
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 2. Generate example data (including various data types)
# Create a realistic-looking dataset with mixed numerical and categorical features.
data = {
    "age": np.random.randint(18, 65, 100),
    "fare": np.random.uniform(50, 500, 100),
    "embarked": np.random.choice(["S", "C", "Q"], 100, p=[0.7, 0.2, 0.1]),
    "sex": np.random.choice(["male", "female"], 100, p=[0.6, 0.4]),
    "target": np.random.randint(0, 2, 100),  # Target variable for binary classification
}
df = pd.DataFrame(data)
print(df)
X = df.drop("target", axis=1)
print(X)
y = df["target"]
print(y)

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Sample of the original dataset:")
print(X_train.head())
print("-" * 50)


# ==============================================================================
# Section 1: Building the Preprocessing Pipeline using ColumnTransformer
# Define appropriate preprocessors for each data type and integrate them with ColumnTransformer.
# ==============================================================================

# Separate numerical and categorical features
numerical_features = ["age", "fare"]
categorical_features = ["embarked", "sex"]

# (1) Pipeline for numerical data preprocessing (Scaling)
# Use StandardScaler.
numerical_pipeline = Pipeline([("scaler", StandardScaler())])

# (2) Pipeline for categorical data preprocessing (One-Hot Encoding)
# Use drop='first' to prevent multicollinearity.
categorical_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

# (3) Integrate the pipelines above using ColumnTransformer
# Apply scaler to 'age' and 'fare'
# Apply one-hot encoder to 'embarked' and 'sex'
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
    ],
    remainder="passthrough",  # Keep columns not specified above as they are (Not applicable in this example)
)

print("Preprocessing pipeline including ColumnTransformer configured.")
print("-" * 50)

# ==============================================================================
# Section 2: Integrating the entire workflow into a Pipeline
# Bundle the preprocessing step and the final model into a single object.
# First Step: Preprocessing using ColumnTransformer
# Second Step: Final classification model
# ==============================================================================
full_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

# A single call to Pipeline.fit() automates all preprocessing and model training.
print("Starting full pipeline training...")
full_pipeline.fit(X_train, y_train)
print("Full pipeline training complete!")
print("-" * 50)


# ==============================================================================
# Section 3: Prediction and Evaluation using the Pipeline
# ==============================================================================
# A single call to Pipeline.predict() automates both preprocessing and prediction.
y_pred = full_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test dataset accuracy: {accuracy:.4f}")
print("Using a Pipeline, the preprocessing and model training steps are executed automatically and sequentially.")

# Cross-validation using the Pipeline
scores = cross_val_score(full_pipeline, X, y, cv=5, scoring="accuracy")
print(f"\n5-Fold cross-validation scores using the Pipeline: {scores}")
print(f"Cross-validation mean accuracy: {np.mean(scores):.4f}")
print("-" * 50)

"""
Sample of the original dataset:
    age        fare embarked     sex
11   22  101.817375          C  female
47   54  318.720087          C  female
85   34   85.674513          S  female
28   53   87.227896          S  female
93   58  368.277939          S  female
--------------------------------------------------
Preprocessing pipeline including ColumnTransformer configured.
--------------------------------------------------
Starting full pipeline training...
Full pipeline training complete!
--------------------------------------------------
Test dataset accuracy: 0.5000
Using a Pipeline, the preprocessing and model training steps are executed automatically and sequentially.

5-Fold cross-validation scores using the Pipeline: [0.55 0.3  0.45 0.55 0.45]
Cross-validation mean accuracy: 0.4600
--------------------------------------------------
"""

"""
Review

Q1. What problems can arise when applying multiple preprocessors (e.g., StandardScaler, OneHotEncoder) individually without using a ColumnTransformer?

    **Increased Code Complexity**:
        You must manually call `fit_transform` and `transform` for each group of columns. This makes the code longer and reduces readability.
    **Risk of Errors**:
        You must remember the statistics (mean, max value, etc.) from the training data and apply them to the test data, a process prone to human error.
    **Data Leakage Risk**:
        Especially during cross-validation, `fit` must be performed only on the training data of each fold. Manual handling increases the risk of **Data Leakage**, where test data information contaminates the training process.

Q2. In the demo code, what is the role of each element in the tuple `('num', numerical_pipeline, numerical_features)` passed to the `transformers` parameter of `preprocessor = ColumnTransformer(...)`?

    **'num'**:
        This is the **name** of the transformer. A unique name is assigned to later access this step within the pipeline or during hyperparameter tuning.

    **numerical_pipeline**:
        This is the **Transformer instance** to be applied to these columns. In the demo, it's a Pipeline object that includes StandardScaler.

    **numerical_features**:
        This is the **list of columns** to which this transformer will be applied. In the demo, this corresponds to `['age', 'fare']`.

Q3. What is the greatest advantage of combining ColumnTransformer and Pipeline?

    **Single Workflow**:
        The complex preprocessing and model training can all be executed with a single call to `full_pipeline.fit()`, making the code very concise.

    **Automation and Stability**:
        The preprocessing steps are executed automatically and sequentially, preventing human errors that can occur when transforming data manually.

    **Data Leakage Prevention**:
        The Pipeline automatically performs `fit` only on the training data of each fold during cross-validation, fundamentally preventing test data information from contaminating the training process.

Q4. When the Pipeline is used for cross-validation (`cross_val_score`), how does the ColumnTransformer operate in each fold?

    For each fold generated during cross-validation, the Pipeline internally **re-initializes and trains** the entire pipeline object, including the ColumnTransformer.

    This means that for every fold, the ColumnTransformer calls the `fit` method **only on the training data** of that fold (equivalent to X_train) to calculate statistics (mean, standard deviation, etc.).

    Subsequently, only `transform` is applied to the test data using the statistics obtained from the training data. This process ensures strict cross-validation, guaranteeing that test data information does not influence the training.
"""
