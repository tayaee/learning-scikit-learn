"""
This feature involves saving a trained model to a file and loading it later for use or deployment. It allows for efficient model use without needing to retrain the model every time.
    joblib or pickle: Use joblib.dump() to save the model to a file and joblib.load() to load it back.

joblib is a library optimized for efficiently saving and loading large objects, particularly those containing NumPy arrays, like scikit-learn models.
It is essential for model deployment and significantly saves time and resources by allowing the reuse of a saved model file without needing to retrain it after initial training.
"""

# 1. Import necessary libraries
import joblib  # Used for saving/loading models
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 2. Generate example data and train the model
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create model instance and train
model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
print("Starting model training...")
model.fit(X_train, y_train)
print("Model training complete!")

# Check the performance of the trained model on test data
initial_y_pred = model.predict(X_test)
initial_accuracy = accuracy_score(y_test, initial_y_pred)
print(f"Test data accuracy immediately after training: {initial_accuracy:.4f}")
print("-" * 50)


# ==============================================================================
# Section 1: Saving the trained model to a file
# The joblib.dump() function is used to serialize the model object to a file.
# ==============================================================================
print("=== [Section 1] Saving the Model ===")

# Define the filename for saving (the extension .joblib or .pkl is typically used)
model_filename = "trained_svc_model.joblib"

# joblib.dump(object to save, filename)
joblib.dump(model, model_filename)
print(f"Model successfully saved to the file '{model_filename}'.")
print("-" * 50)


# ==============================================================================
# Section 2: Loading and using the saved model file
# The joblib.load() function is used to deserialize the saved model object back into memory.
# ==============================================================================
print("=== [Section 2] Loading and using the Model ===")

# joblib.load(filename)
# Load the saved model into a new variable.
loaded_model = joblib.load(model_filename)

# Re-check the prediction performance of the loaded model
# The model can be used immediately for prediction without needing to be retrained.
loaded_y_pred = loaded_model.predict(X_test)
loaded_accuracy = accuracy_score(y_test, loaded_y_pred)

print(f"Loaded model's test data accuracy: {loaded_accuracy:.4f}")

# Verify that the accuracy is the same before and after saving
print(f"Accuracy consistent: {initial_accuracy == loaded_accuracy}")
print("-" * 50)

# Tip:
# The compress option in joblib.dump() can be used to reduce file size.
joblib.dump(model, "compressed_model.joblib", compress=3)

"""
Starting model training...
Model training complete!
Test data accuracy immediately after training: 0.8333
--------------------------------------------------
=== [Section 1] Saving the Model ===
Model successfully saved to the file 'trained_svc_model.joblib'.
--------------------------------------------------
=== [Section 2] Loading and using the Model ===
Loaded model's test data accuracy: 0.8333
Accuracy consistent: True
--------------------------------------------------
"""

"""
Q1. What is the main purpose of **Model Serialization**, and why is it important in a machine learning workflow?

    The main purpose of model serialization is to **persistently save the object state of a trained machine learning model** by converting it into a file.
    This is important for the following reasons:
        **Reusability**: A model trained once can be loaded and used immediately for prediction without the need for retraining every time it's needed.
        **Deployment**: The trained model can be easily integrated and deployed into a web service or application.
        **Time and Resource Savings**: It saves the substantial time and computing resources required to train models on large datasets.

Q2. What are the roles of the `joblib.dump()` and `joblib.load()` functions, and what are their parameters?

    **joblib.dump(value, filename, compress=0, ...)**:
        **Role**: Serializes and saves a Python object (`value`) to a file (`filename`).
        **Key Parameters**:
            `value`: The Python object to be saved (e.g., a trained scikit-learn model).
            `filename`: The path and name of the file to save the object to.

    **joblib.load(filename, mmap_mode=None)**:
        **Role**: Reads the serialized object from the specified file (`filename`) and restores it into memory (deserialization).
        **Key Parameters**:
            `filename`: The path and name of the file to load.

Q3. In the demo code, why did the model achieve the same accuracy without being retrained after using joblib?

    `joblib.dump()` saves the **entire state** of the trained model object exactly as it is to the file. This includes not only the model's hyperparameters (kernel, C, gamma) but also all the information necessary for prediction, such as the **weights** or **support vectors** (in the case of SVC) determined during the training process.

    When `joblib.load()` is used to restore this file, the saved model state is **perfectly reconstructed**. Therefore, the model exhibits the same prediction performance as it did immediately after initial training, without the need for retraining.

Q4. Why is joblib generally preferred over pickle for saving scikit-learn models?

    **Efficiency**:
        joblib is significantly **faster and more efficient** than pickle, especially when dealing with large objects that contain NumPy arrays. Since scikit-learn models rely heavily on NumPy arrays internally, joblib is optimized for them.

    **Memory Mapping**:
        joblib supports **memory mapping**, which allows large model files to be read directly from the disk as needed without loading the entire file into memory at once. This is a significant advantage for minimizing memory usage.

    **Safety**:
        Pickle can pose security risks with files received from untrusted sources. In comparison, joblib is generally considered to be **relatively safer** in this regard.
"""
