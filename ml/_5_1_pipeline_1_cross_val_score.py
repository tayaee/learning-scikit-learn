from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Example Data
X, y = load_iris(return_X_y=True)

# Pipeline: [Scaler, Classifier]
pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))])

# 5-fold cross-validation with cross_val_score
scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")

print("Cross-validation accuracy for each fold:", scores)
print("Mean cross-validation accuracy: {:.4f}".format(scores.mean()))
