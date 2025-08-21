from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 예제 데이터
X, y = load_iris(return_X_y=True)

# Pipeline: [스케일러, 분류기]
pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))])

# cross_val_score로 5-폴드 교차 검증
scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")

print("각 폴드의 교차 검증 정확도:", scores)
print("평균 교차 검증 정확도: {:.4f}".format(scores.mean()))
