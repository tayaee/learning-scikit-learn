import numpy as np
from scipy.sparse import csr_matrix
from pyfm import pylibfm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ----------------------------------------------------
# 1. Setup and Data Preparation (Sparse Matrix Format)
# ----------------------------------------------------
# In real-world recommender systems, data is typically sparse after
# Label and One-Hot Encoding.

# Example: 10 samples with 5 features in a sparse matrix format (CSR)

# Data (values, row indices, column indices)
data = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
row_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3]
col_ind = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 2, 3, 4, 0]

# Target Labels (Classification: 0 or 1)
target = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

# Create the Compressed Sparse Row (CSR) matrix
X = csr_matrix((data, (row_ind, col_ind)), shape=(10, 5))

print("--- Sparse Input Matrix (CSR) ---")
print(X.toarray())
print(f"Target Labels: {target}")

# 2. Split Data into Training and Testing Sets
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.3, random_state=42)

# 3. Build and Train the FM Model
# ----------------------------------------------------
# Instantiate the pylibfm.FM model object
# num_factors=10: Sets the dimensionality (K) of the latent vectors.
# num_iter=10: Sets the number of iterations (epochs).
# task='classification': Specifies a classification problem (use 'regression' for regression).
fm_model = pylibfm.FM(num_factors=10, num_iter=10, task="classification", verbose=True, seed=42)

print("\n--- Starting FM Model Training ---")
# FM Training: Learns linear weights and latent vectors from the sparse matrix
fm_model.fit(X_train, y_train)
print("--- FM Model Training Complete ---")


# 4. Prediction and Evaluation
# ----------------------------------------------------
# Generate prediction probabilities (values between 0 and 1) on the test data
y_pred_proba = fm_model.predict(X_test)

# Evaluate performance using the ROC-AUC score (suitable for classification)
auc_score = roc_auc_score(y_test, y_pred_proba)

print("\n[FM Model Performance Evaluation]")
print(f"Predicted Probabilities (First 3): {y_pred_proba[:3]}")
print(f"ROC AUC Score: {auc_score:.4f}")
