"""
Supervised Learning:
    Classification: Used to classify data into pre-defined classes or categories.
        Examples: sklearn.svm.SVC, sklearn.ensemble.RandomForestClassifier

SVC (Support Vector Classifier):
    SVC (Support Vector Classifier) is a classification model that uses the Support Vector Machine algorithm.
    This model focuses on finding the optimal Decision Boundary that best separates the data points.
    It demonstrates powerful performance, especially in effectively classifying non-linear data by transforming it into a high-dimensional space using the 'Kernel Trick'.

Key Concepts:
    Hyperplane
    Margin
    Support Vectors
    Kernel
"""

# 1. Import necessary libraries
from sklearn.datasets import make_classification  # Generate virtual data for classification
from sklearn.metrics import accuracy_score, classification_report  # Model evaluation metrics
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.svm import SVC  # Support Vector Classifier (SVC) - Supervised Learning Classification Model

# 2. Generate example data
# Create virtual classification data using the make_classification function.
# For example, we will simulate a 'cancer diagnosis' scenario.
# - n_samples=100: Data for 100 patients
# - n_features=2: 2 examination metrics (e.g., Feature1='Tumor Size', Feature2='Cell Shape Irregularity')
# - n_classes=2: 2 diagnosis results (e.g., Class0='Benign', Class1='Malignant')
# - n_redundant=0: No redundant features (features created by a combination of other features)
# - random_state=42: Seed value for reproducible results
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_redundant=0, random_state=42)

# 3. Data Splitting
# Split the data into training (train) and test sets.
# The training set is used for model training, and the test set is used for model performance evaluation.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Create Estimator Instance
# SVC stands for 'Support Vector Classifier', an estimator used for classification problems.
# C: Regularization parameter. A smaller C value means stronger regularization and a simpler model.
# kernel: The function used to classify the data. Examples include 'linear' and 'rbf'.
# 'rbf' is the most commonly used and is effective for non-linear data.
model = SVC(kernel="linear", C=1.0, random_state=42)

# 5. Model Training (fit)
# Use the model.fit(X, y) method to fit the model to the training data.
# The model learns the data patterns and finds the optimal decision boundary through this process.
model.fit(X_train, y_train)

# 6. Prediction (predict)
# Use the model.predict(X) method to perform predictions on the test data using the trained model.
# The prediction results will be class labels, such as 0 or 1.
y_pred = model.predict(X_test)

# 7. Model Evaluation
# Evaluate the model's performance by comparing the actual values (y_test) with the predicted values (y_pred).

# Accuracy: The ratio of correct predictions out of all predictions.
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Classification Report: Provides more detailed evaluation metrics like Precision, Recall, and F1-score.
# Precision: The ratio of actual 'positives' among those predicted as 'positive'.
# Recall: The ratio of 'positives' correctly predicted as 'positive' among all actual 'positives'.
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

"""
Model Accuracy: 1.00

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        16
           1       1.00      1.00      1.00        14

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
"""

"""
Review

Q1. What type of machine learning problem is the sklearn.svm.SVC model used for?

    - SVC is used for **Classification** problems, which are a type of **Supervised Learning**.
      It is utilized to classify given data into one of several predefined classes.

Q2. What is the role of the fit() method in the code, and what data does it take as arguments?

    - The **fit()** method serves to train the model.
      It takes the **feature data (X_train)** and the **correct labels (y_train)** corresponding to those features as arguments, allowing the model to learn the data patterns.

Q3. What is the purpose of the predict() method, and what is its output?

    - The **predict()** method is used to make predictions on new data (i.e., data not used for training) using the trained model.
      Its output is an array of **class labels** predicted by the model for the input data.

Q4. Why is train_test_split used? What problem can arise if this step is omitted?

    - **train_test_split** is used to prevent **overfitting** of the model and to objectively evaluate the model's generalization performance.

    - If this step is omitted and the model is trained and evaluated on the entire dataset, the model may become overly tailored to the training data,
      leading to **overfitting** where performance suffers on actual new, unseen data.
      Using train_test_split allows the model's true performance to be measured using test data it has never seen.

Q5. What aspects of the model are assessed by accuracy_score and classification_report, respectively?

    - **accuracy_score** is used to assess **accuracy**, which is the ratio of correct predictions out of all predictions made by the model.

    - **classification_report** provides more fine-grained performance metrics than just accuracy, such as **Precision**, **Recall**, and **F1-score** for each class,
      which helps in understanding the model's strengths and weaknesses.

Q6. What are the key differences between RandomForestClassifier and SVC?

    Performance Comparison: Which algorithm is superior depends on the characteristics of the dataset.

        RF's Advantages: It shows excellent performance with large datasets, high-dimensional data (many features), and data with many complex non-linear relationships.
        It also tends to perform well without extensive data preprocessing (scaling, normalization), making it convenient to use.

        SVC's Advantages: It performs exceptionally well when the number of features (dimensions) is greater than the number of samples, or when the dataset size is small to moderate.
        It achieves high accuracy by finding a powerful 'optimal hyperplane', especially in data where the boundaries between classes are clear.

    Reputation Comparison:

        RF: It has a strong reputation as a "first algorithm to try" in modern machine learning.
        It provides stable performance without special tuning and is versatile for both classification and regression problems.
        It is a trusted model as a leading example of ensemble learning.

        SVC: It has long been considered a 'classic powerhouse'.
        Its strength has been proven in high-dimensional classification problems such as bioinformatics (gene classification) and image recognition.
        However, it can be computationally expensive (slow training time) with large datasets, sometimes leading to it being overlooked in favor of more efficient algorithms.

    Comparison of Main Application Areas:

        Random Forest (RF)
            Finance: Credit card fraud detection, stock market trend prediction.
            Healthcare: Disease diagnosis (e.g., cancer diagnosis), genomic data analysis.
            E-commerce: Product recommendation systems, customer churn prediction.
            Others: Widely used in complex and large-scale data applications like land use classification via satellite image analysis.

        SVC (Support Vector Classifier)
            Bioinformatics: Genomic data classification and pattern recognition.
            Natural Language Processing: Text classification, spam filtering.
            Computer Vision: Face recognition, image classification.
            Others: Particularly useful for handling high-dimensional data with clear class boundaries, such as handwriting recognition and signal processing.

    Conclusion:

        RF:
            A good choice when dealing with **large datasets**,
            when **rapid prototyping** is necessary,
            and when some degree of **model interpretability** is required.

        SVC:
            Shows very powerful performance when the dataset size is **small to moderate**,
            when dealing with **high-dimensional data**,
            and when finding a **clear decision boundary** is crucial.

Q7. What are the roles of SVC's main hyperparameters: kernel, C, and gamma?

    The performance of the SVC model largely depends on how these three hyperparameters are configured.

    1. **kernel**:
        - **Role**: The 'tool' that determines what type of decision boundary should be drawn to learn the data patterns.

        - **Main Values**:
            - `'rbf'` (default): The **Radial Basis Function kernel, effective for non-linear data.**
              It can create complex boundaries.
            - `'linear'`: A linear kernel that separates data with a straight line (hyperplane).
              It is fast and effective when the data is linearly separable.

    2. **C (Cost, Regularization Parameter)**:
        - **Role**: Controls the **strength of regularization**, which determines how many errors the model is willing to tolerate.

        - **Larger Value (e.g., 100)**: Weaker regularization.
          The model attempts to tolerate few errors on the training data, leading to a complex decision boundary and potential **Overfitting**.

        - **Smaller Value (e.g., 0.1)**: Stronger regularization.
          The model allows some errors to achieve a wider margin, resulting in a simpler decision boundary and potentially better generalization performance.

    3. **gamma**:
        - **Role**: In non-linear kernels like `rbf`, it determines the **range of influence** that a single data point has on the decision boundary.

        - **Larger Value**: The range of influence is narrow, making the model highly sensitive to each data point.
          The decision boundary becomes very convoluted, leading to potential **Overfitting**.

        - **Smaller Value**: The range of influence is broad, leading to a very smooth and simple decision boundary and potential **Underfitting**.

    Since these three parameters interact with each other, it is standard practice to use tools like `GridSearchCV` to find the optimal combination.
"""
