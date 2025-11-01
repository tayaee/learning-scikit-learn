"""
Model Management
    This goes beyond just saving a trained model to a file; it involves saving the model's information (metadata) alongside it to enable **systematic management**, **easy searching**, and **reuse** when needed.
    This is an important first step in MLOps.
"""

# 1. Import necessary libraries
import datetime
import json
import os
import uuid

import joblib
import sklearn
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def save_model_with_metadata(model, metrics, model_dir="models"):
    """
    Saves the trained model along with its relevant metadata.

    Args:
        model: The trained scikit-learn model object.
        metrics (dict): The model's performance indicators (e.g., {'accuracy': 0.95}).
        model_dir (str): The directory to save the model and metadata.

    Returns:
        str: The unique ID of the saved model.
    """
    # Create the saving directory if it does not exist
    os.makedirs(model_dir, exist_ok=True)

    # 1. Generate model identification information
    model_id = str(uuid.uuid4())
    model_filename = f"{model_id}.joblib"
    metadata_filename = f"{model_id}.json"
    model_filepath = os.path.join(model_dir, model_filename)
    metadata_filepath = os.path.join(model_dir, metadata_filename)

    # 2. Save the model file (Serialization)
    joblib.dump(model, model_filepath)
    print(f"Model saved to path: '{model_filepath}'.")

    # 3. Create the metadata dictionary
    metadata = {
        "model_uuid": model_id,
        "model_filepath": model_filepath,
        "model_class": model.__class__.__name__,
        "hyperparameters": model.get_params(),
        "metrics": metrics,
        "timestamp": datetime.datetime.now().isoformat(),
        "sklearn_version": sklearn.__version__,
        "description": "SVC model for binary classification demo.",
    }

    # 4. Save the metadata JSON file
    with open(metadata_filepath, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to path: '{metadata_filepath}'.")

    return model_id


# ==============================================================================
# Section 1: Model Training and Saving with Metadata
# ==============================================================================
print("=== [Section 1] Model Training and Saving ===")

# Generate example data and train the model
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
model_metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
}

# Use the function to save the model and metadata
saved_model_id = save_model_with_metadata(model, model_metrics)
print(f"Saved Model ID: {saved_model_id}")
print("-" * 50)


# ==============================================================================
# Section 2: Loading the Model using Metadata
# ==============================================================================
print("\n=== [Section 2] Model Loading and Usage using Metadata ===")

# Path to the saved metadata file
metadata_path_to_load = os.path.join("models", f"{saved_model_id}.json")

# 1. Load the metadata
with open(metadata_path_to_load, "r") as f:
    loaded_metadata = json.load(f)

print("Loaded Metadata:")
print(json.dumps(loaded_metadata, indent=4))

# 2. Find the model file path from the metadata and load the model
model_path_to_load = loaded_metadata["model_filepath"]
loaded_model = joblib.load(model_path_to_load)

# 3. Perform prediction with the loaded model
new_prediction = loaded_model.predict(X_test[:1])
print(f"\nPrediction result for new data using the loaded model: {new_prediction}")
print("-" * 50)
