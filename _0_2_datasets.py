import matplotlib.pyplot as plt
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    make_blobs,
    make_circles,
    make_classification,
    make_moons,
    make_regression,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# 2개의 유의미한 특징을 가진 2클래스 분류용 데이터 100개 생성
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42,
)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolor="k")
plt.title("make_classification Example")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# ---------------------------------------------------------
# 1개의 특징을 가진 회귀용 데이터 100개 생성
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, edgecolor="k")
plt.title("make_regression Example")
plt.xlabel("Feature")
plt.ylabel("Target Value")
plt.show()


# ---------------------------------------------------------
# 3개의 군집을 가진 데이터 200개 생성
X, y = make_blobs(n_samples=200, centers=3, cluster_std=1.0, random_state=42)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolor="k")
plt.title("make_blobs Example")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# ---------------------------------------------------------
# 초승달 모양 데이터 200개 생성
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolor="k")
plt.title("make_moons Example")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# ---------------------------------------------------------
# 동심원 모양 데이터 200개 생성
X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolor="k")
plt.title("make_circles Example")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# ---------------------------------------------------------
# 설명: 붓꽃의 종류를 분류하는 가장 대표적인 데이터셋입니다. 3개의 클래스(Setosa, Versicolour, Virginica)와 4개의 특징(꽃받침/꽃잎의 길이/너비)으로 구성됩니다.
# 용도: 다중 클래스 분류(Multi-class classification) 알고리즘의 기본 예제로 사용됩니다.
iris = load_iris()

# 2. 데이터 확인
X = iris.data
y = iris.target

print("--- 붓꽃(Iris) 데이터셋 ---")
print(f"데이터의 형태: {X.shape}")
print(f"타겟의 형태: {y.shape}")
print(f"특징 이름: {iris.feature_names}")
print(f"타겟 이름: {iris.target_names}")  # type: ignore
# print("\n데이터셋 설명:\n", iris.DESCR) # 주석 해제 시 전체 설명 확인 가능
print("-" * 50)


# ---------------------------------------------------------
# 설명: 0부터 9까지의 손글씨 이미지(8x8 픽셀)를 64차원 벡터로 변환한 데이터입니다.
# 용도: 이미지 분류, 차원 축소(PCA, t-SNE) 알고리즘 테스트에 널리 사용됩니다.
digits = load_digits()

# 2. 데이터 확인
X = digits.data
y = digits.target

print("--- 손글씨 숫자(Digits) 데이터셋 ---")
print(f"데이터의 형태: {X.shape}")
print(f"타겟의 형태: {y.shape}")

# 3. 첫 번째 이미지 데이터 시각화
plt.figure(figsize=(2, 2))
plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation="nearest")
plt.title(f"Label: {digits.target[0]}")
plt.show()
print("-" * 50)


# ---------------------------------------------------------
# 설명: 유방암 진단 결과를 악성(malignant)과 양성(benign)으로 분류하는 데이터셋입니다. 30개의 다양한 의학적 특징을 포함합니다.
# 용도: 이진 분류(Binary classification) 모델의 성능 평가 및 하이퍼파라미터 튜닝 예제로 자주 사용됩니다.
cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

print("--- 유방암(Breast Cancer) 데이터셋 ---")
print(f"데이터의 형태: {X.shape}")
print(f"타겟의 형태: {y.shape}")
print(f"특징 이름 (일부): {cancer.feature_names[:5]}")
print(f"타겟 이름: {cancer.target_names}")  # type: ignore
print("-" * 50)


# ---------------------------------------------------------
# 설명: 당뇨병 환자들의 10가지 특징(나이, 성별, BMI, 혈압 등)을 바탕으로 1년 후의 질병 진행도를 예측하는 데이터셋입니다.
# 용도: 선형 회귀(Linear Regression), 릿지(Ridge), 라쏘(Lasso) 등 다양한 회귀 모델을 테스트하는 데 사용됩니다.
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

print("--- 당뇨병(Diabetes) 데이터셋 ---")
print(f"데이터의 형태: {X.shape}")
print(f"타겟의 형태: {y.shape}")
print(f"특징 이름: {diabetes.feature_names}")
print("-" * 50)


# ---------------------------------------------------------
# as_frame=True로 설정하면 데이터를 바로 pandas DataFrame 형태로 받을 수 있어 편리합니다.
housing = fetch_california_housing(as_frame=True)

# housing 객체는 데이터(frame), 타겟(target), 특징 이름(feature_names) 등을 포함합니다.
X = housing.data
y = housing.target

print("--- 캘리포니아 주택 가격(California Housing) 데이터셋 ---")
print(f"데이터의 형태: {X.shape}")
print(f"타겟의 형태: {y.shape}")
print(f"특징 이름: {housing.feature_names}")
# print("\n데이터셋 설명:\n", housing.DESCR) # 주석 해제 시 전체 설명 확인 가능

# DataFrame으로 데이터 확인 (상위 5개)
print("\n--- 데이터프레임 확인 (상위 5개) ---")
# as_frame=True로 로드했으므로 X가 이미 DataFrame입니다.
# 타겟 변수(y)를 합쳐서 전체 데이터를 살펴봅니다.
df = X.copy()
df["MedHouseVal"] = y
print(df.head())
print("-" * 50)


# 4. 간단한 회귀 모델 학습 및 평가
print("\n--- 선형 회귀 모델 학습 및 평가 예제 ---")

# 데이터를 학습 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)
print("선형 회귀 모델 학습 완료!")

# 테스트 데이터로 예측 수행
y_pred = model.predict(X_test)

# 모델 성능 평가
mse = mean_squared_error(y_test, y_pred)
r2 = model.score(X_test, y_test)

print(f"\n테스트 데이터에 대한 평균 제곱 오차 (MSE): {mse:.4f}")
print(f"테스트 데이터에 대한 결정 계수 (R-squared): {r2:.4f}")
print("-" * 50)
