"""
지금까지 다룬 모델 외에도 scikit-learn에는 다양한 알고리즘이 있습니다.
    앙상블 모델: GradientBoostingClassifier, XGBoost, LightGBM 등 RandomForest보다 더 강력한 성능을 내는 모델들이 있습니다. (LightGBM과 XGBoost는 별도 설치 필요)
    커널 기반 모델: SVC 외에도 SVR (회귀)과 같은 커널 기반 모델들이 있습니다.
    선형 모델: LinearRegression 외에 Lasso, Ridge와 같이 규제(Regularization)를 통해 과적합을 방지하는 모델들이 있습니다.

Lasso (Least Absolute Shrinkage and Selection Operator):
    Lasso는 L1 규제가 적용된 선형 회귀 모델입니다.
    일반적인 선형 회귀 모델에 규제 항을 추가하여 모델의 복잡도를 낮추고 과적합(Overfitting)을 방지하는 데 사용됩니다.
    특히, L1 규제는 불필요한 특징(Feature)의 계수(coefficient)를 0으로 만들어버리는 특징 때문에
    특징 선택(Feature Selection) 효과도 함께 얻을 수 있습니다.
"""

# 1. 필요한 라이브러리 임포트
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 2. 예제 데이터 생성
# '주택 가격'을 예측하는 비즈니스 시나리오를 모사합니다.
# 100개의 주택 데이터(n_samples)와 5개의 특징(n_features)을 가진 데이터셋을 만듭니다.
#
# - y (타겟 변수): '주택 가격'
# - 유의미한 특징 (가격에 영향을 줌):
#   - X[:, 0]: '집 크기(m²)' (가장 큰 영향, 계수=3)
#   - X[:, 1]: '방 개수' (중간 영향, 계수=2)
#   - X[:, 2]: '지하철역과의 거리' (작은 영향, 계수=1)
# - 불필요한 특징 (가격과 상관관계 없음):
#   - X[:, 3]: '지역 공원 방문객 수'
#   - X[:, 4]: '인근 상점의 일일 평균 매출'
#
# Lasso 모델이 이 불필요한 특징(x4, x5)들의 계수를 0으로 만들어,
# 자동으로 중요한 특징만 선택하는지 확인하는 것이 이 데모의 목표입니다.
np.random.seed(42)
X = np.random.randn(100, 5)
y = 3 * X[:, 0] + 2 * X[:, 1] + 1 * X[:, 2] + np.random.randn(100) * 2

# 주석에 대략 일치하는 가짜 데이터 셋을 만듭니다.
# 유의미한 특징 (가격에 영향을 줌):
#   - X[:, 0]: '집 크기(m²)' (가장 큰 영향, 계수=3)
#   - X[:, 1]: '방 개수' (중간 영향, 계수=2)
#   - X[:, 2]: '지하철역과의 거리' (작은 영향, 계수=1)
# X = np.random.rand(100, 5) * np.array([100, 10, 1, 1000, 100])  # 스케일 차이 부여
# y = 3 * X[:, 0] + 2 * X[:, 1] + 1 * X[:, 2] + np.random.randn(100) * 5  # 노이즈 추가

# 생성된 데이터의 통계 정보 확인
# np.random.randn()은 표준 정규 분포(평균 0, 표준편차 1)를 따르는 데이터를 생성합니다.
# 실제 생성된 데이터의 평균과 표준편차는 0과 1에 매우 가깝습니다.
print("\n--- 생성된 데이터(X)의 통계 정보 ---")
print(f"전체 특징(X)의 평균: {X.mean():.4f}")
print(f"전체 특징(X)의 표준편차: {X.std():.4f}")
print("-" * 50)

# 3. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==============================================================================
# 섹션 1: 일반 선형 회귀 (Lasso 미적용)
# 규제가 없는 기본 선형 회귀 모델의 계수를 확인합니다.
# ==============================================================================
print("=== [Section 1] 일반 선형 회귀 모델 ===")
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred_lin = lin_reg.predict(X_test)
mse_lin = mean_squared_error(y_test, y_pred_lin)

print(f"일반 선형 회귀 모델의 계수: {lin_reg.coef_}")
print(f"MSE (오차): {mse_lin:.4f}")
print("-" * 50)


# ==============================================================================
# 섹션 2: Lasso 모델 적용 (L1 규제)
# Lasso 모델의 규제 효과와 특징 선택 효과를 확인합니다.
# ==============================================================================
print("=== [Section 2] Lasso 모델 ===")

# Lasso 모델 인스턴스 생성
# alpha: 규제 강도를 조절하는 하이퍼파라미터.
#        alpha가 클수록 규제가 강해져 계수들이 0에 가까워집니다.
lasso_reg = Lasso(alpha=1.0, random_state=42)
lasso_reg.fit(X_train, y_train)

y_pred_lasso = lasso_reg.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

print(f"Lasso 모델의 계수 (alpha=1.0): {lasso_reg.coef_}")
print(f"MSE (오차): {mse_lasso:.4f}")

# 불필요한 특징(4번째, 5번째 특징)의 계수가 0에 가깝거나 0이 되었음을 주목하세요.
print("\n- 불필요한 특징의 계수가 0에 수렴하여 특징 선택 효과를 보입니다.")
print("-" * 50)


# ==============================================================================
# 섹션 3: alpha 값에 따른 계수 변화 시각화
# alpha 값에 따라 계수가 어떻게 변하는지 시각적으로 확인합니다.
# ==============================================================================
print("=== [Section 3] alpha 값에 따른 계수 변화 ===")
# np.logspace(-2, 2, 100)는 10^-2 (0.01)부터 10^2 (100)까지의 범위를
# 로그 스케일로 균등하게 나눈 100개의 값을 생성합니다.
# 이를 통해 alpha 값이 작을 때와 클 때의 변화를 모두 효과적으로 관찰할 수 있습니다.
# 이 코드는 총 100개의 alpha 값을 생성합니다.
alphas = np.logspace(-2, 2, 100)
coefs = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)

plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale("log")
plt.xlabel("Alpha (Regularization Strength)")
plt.ylabel("Coefficients")
plt.title("Lasso Coefficients as a function of Alpha")
plt.legend([f"Feature {i + 1}" for i in range(5)])
plt.axis("tight")
plt.grid(True)
plt.show()

"""
실행 예시

--- 생성된 데이터(X)의 통계 정보 ---
전체 특징(X)의 평균: 0.0364
전체 특징(X)의 표준편차: 0.9945
--------------------------------------------------
=== [Section 1] 일반 선형 회귀 모델 ===
일반 선형 회귀 모델의 계수: [3.1631208  2.13159999 1.06827716 0.08499474 0.00957412]        
MSE (오차): 3.2356
--------------------------------------------------
=== [Section 2] Lasso 모델 ===
Lasso 모델의 계수 (alpha=1.0): [ 1.69950905  1.03821904  0.0140734  -0.         -0.        ]
MSE (오차): 7.3102

- 불필요한 특징의 계수가 0에 수렴하여 특징 선택 효과를 보입니다.
--------------------------------------------------
=== [Section 3] alpha 값에 따른 계수 변화 ===
"""

"""
복습

Q1. Lasso 모델에서 규제(Regularization)가 하는 역할은 무엇이며, 왜 규제가 과적합(Overfitting)을 방지하는 데 도움이 되나요?

    - 규제는 모델이 데이터에 너무 과도하게 맞춰지는 것을 방지하기 위해 
      **모델의 복잡도에 패널티(penalty)**를 주는 역할을 합니다. 
      Lasso는 모델의 계수들의 절댓값 합에 비례하여 패널티를 부여합니다.

    - 과적합된 모델은 보통 특정 특징에 대한 계수 값이 매우 커지는데, 
      규제는 이 계수들을 억제하여 모델을 더 단순하고 일반적인 형태로 만들어 과적합을 방지하는 데 도움이 됩니다.

Q2. Lasso와 일반 선형 회귀의 가장 큰 차이점은 무엇이며, Lasso가 '특징 선택(Feature Selection)'에 유용하다고 불리는 이유를 설명해 주세요.

    - 가장 큰 차이점: Lasso는 손실 함수(Loss Function)에 L1 규제 항을 추가하는 반면, 일반 선형 회귀는 규제 항이 없습니다.

    - 특징 선택: Lasso의 L1 규제는 패널티가 너무 커지면 계수 값을 아예 0으로 만들어버리는 성질이 있습니다. 
      이는 모델이 불필요하거나 중요하지 않은 특징을 자동으로 제외시키는 효과를 가져오며, 이 때문에 Lasso는 특징 선택에 유용하다고 평가받습니다.

Q3. Lasso 모델의 하이퍼파라미터인 alpha 값의 역할은 무엇이며, alpha가 커질수록 모델은 어떻게 변하나요?

    - alpha는 Lasso 모델의 규제 강도를 조절하는 하이퍼파라미터입니다.

    - alpha 값이 커질수록 규제가 강해집니다. 
      모델의 패널티가 커지므로, 계수 값들이 더 빠르게 0으로 수렴하게 됩니다. 
      즉, 모델이 더 단순해지고 과적합을 방지하는 효과가 강해집니다.

    - alpha 값이 작아질수록 규제가 약해집니다. 
      alpha가 0에 가까워지면 Lasso는 일반 선형 회귀 모델과 거의 동일해집니다.

Q4. 데모 코드의 시각화 결과에서 alpha 값이 커질수록 계수들이 0에 가까워지는 것을 볼 수 있습니다. 이는 어떤 의미인가요?

    - 이 시각화는 alpha라는 규제 강도가 증가함에 따라 모델의 복잡도가 어떻게 감소하는지 보여줍니다.

    - alpha가 작을 때는 모든 특징의 계수가 유의미한 값을 가지지만, 
      alpha가 점차 커지면서 중요도가 낮은 특징(예: 데모 코드의 4, 5번째 특징)의 계수가 먼저 0이 되고, 
      결국에는 중요도가 높은 특징의 계수들까지 0에 가까워지는 것을 볼 수 있습니다. 
      이는 Lasso가 점진적으로 모델을 단순화하고 불필요한 특징을 제거하는 과정을 시각적으로 보여주는 것입니다.

Q5. 그래서 모델 튜닝 방법은?

    - alpha 값을 조정하여 MSE가 최소화되는 최적의 값을 찾는 것이 일반적인 튜닝 방법입니다.

    - alpha 값을 조정하여 r2_score가 최대화되는 최적의 값을 찾는 것도 좋은 방법입니다.
      r^2 = 1 - (MSE / Var(y)) 이므로, MSE가 최소화되면 r2_score가 최대화됩니다.

    GridSearchCV와 같은 기법을 사용하여 여러 alpha 값을 시험해보고, 최적의 alpha 값을 찾는 것이 일반적입니다.
"""
