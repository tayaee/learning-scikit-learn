import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

# 1. Simulate Severe Data Imbalance (e.g., Information Retrieval)
# Total 1000 samples: 99% Negative (0), only 1% Positive (1)
n_samples = 1000
n_positive = 10
n_negative = n_samples - n_positive

# True labels: highly imbalanced
y_true = np.hstack([np.zeros(n_negative), np.ones(n_positive)])

# Simulate Prediction Probabilities:
# A fairly good model (assigns high probability to positive samples)
# (Positive samples get scores between 0.8 and 1.0, Negative samples between 0.0 and 0.2)
scores_positive = np.random.uniform(0.8, 1.0, n_positive)
scores_negative = np.random.uniform(0.0, 0.2, n_negative)
y_scores = np.hstack([scores_negative, scores_positive])

# 2. Calculate ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# 3. Calculate Precision-Recall (PR) Curve and AP (Average Precision)
precision, recall, _ = precision_recall_curve(y_true, y_scores)
# Average Precision (AP) is the metric commonly used to summarize the PR curve area.
average_precision = average_precision_score(y_true, y_scores)

# 4. Visualization
plt.figure(figsize=(14, 6))

# --- ROC Curve Plot ---
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR) - Recall")
plt.title("ROC Curve (Imbalanced Data)")
plt.legend(loc="lower right")
plt.grid(True)
plt.gca().set_aspect("equal", adjustable="box")


# --- Precision-Recall (PR) Curve Plot ---
plt.subplot(1, 2, 2)
# Baseline: Ratio of the positive class (n_positive / n_samples)
baseline = n_positive / n_samples
plt.plot([0, 1], [baseline, baseline], linestyle="--", color="navy", label=f"Baseline Precision ({baseline:.2f})")
plt.plot(recall, precision, color="green", lw=2, label=f"PR curve (AP = {average_precision:.2f})")

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Imbalanced Data)")
plt.legend(loc="upper right")
plt.grid(True)
plt.gca().set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.show()

# Conclusion Message
print("--- Result Analysis ---")
print(f"Dataset Imbalance Ratio: 1 (Positive) : {n_negative / n_positive} (Negative)")
print(f"ROC AUC: {roc_auc:.4f} (Appears excellent, but masks issues with small positive class)")
print(
    f"PR Average Precision (AP): {average_precision:.4f} (More sensitive to performance on the minority positive class)"
)
