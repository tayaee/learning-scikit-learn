"""
지금까지 다룬 모델 외에도 scikit-learn에는 다양한 알고리즘이 있습니다.
    앙상블 모델: GradientBoostingClassifier, XGBoost, LightGBM 등 RandomForest보다 더 강력한 성능을 내는 모델들이 있습니다. (LightGBM과 XGBoost는 별도 설치 필요)
    커널 기반 모델: SVC 외에도 SVR (회귀)과 같은 커널 기반 모델들이 있습니다.
    선형 모델: LinearRegression 외에 Lasso, Ridge와 같이 규제(Regularization)를 통해 과적합을 방지하는 모델들이 있습니다.

Ridge (릿지)는 L2 규제가 적용된 선형 회귀 모델입니다.
    Lasso와 마찬가지로 일반 선형 회귀에 규제 항을 추가하여 모델의 복잡도를 낮추고 과적합(Overfitting)을 방지하는 데 사용됩니다.
    Lasso의 L1 규제가 계수의 절댓값 합에 패널티를 주는 반면,
    Ridge의 L2 규제는 계수의 제곱 합에 패널티를 줍니다.
    이 때문에 계수들을 0에 가깝게 만들지만, 완전히 0으로 만들지는 않습니다.
"""

# 1. 필요한 라이브러리 임포트
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 2. 예제 데이터 생성
# 100개의 샘플과 5개의 특징을 가진 데이터셋을 만듭니다.
# Lasso 예제와 동일하게 불필요한 특징(x4, x5)을 포함하여 Ridge 모델의 작동 방식을 비교합니다.
np.random.seed(42)
X = np.random.randn(100, 5)
y = 3 * X[:, 0] + 2 * X[:, 1] + 1 * X[:, 2] + np.random.randn(100) * 2

# 3. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==============================================================================
# 섹션 1: 일반 선형 회귀 (Ridge 미적용)
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
# 섹션 2: Ridge 모델 적용 (L2 규제)
# Ridge 모델의 규제 효과를 확인합니다.
# ==============================================================================
print("=== [Section 2] Ridge 모델 ===")

# Ridge 모델 인스턴스 생성
# alpha: 규제 강도를 조절하는 하이퍼파라미터.
#        alpha가 클수록 규제가 강해져 계수들이 0에 더 가까워집니다.
ridge_reg = Ridge(alpha=1.0, random_state=42)
ridge_reg.fit(X_train, y_train)

y_pred_ridge = ridge_reg.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

print(f"Ridge 모델의 계수 (alpha=1.0): {ridge_reg.coef_}")
print(f"MSE (오차): {mse_ridge:.4f}")

# 계수들이 0에 가깝지만, Lasso와 달리 완전히 0이 되지는 않았음을 주목하세요.
print("\n- 모든 특징의 계수들이 0에 가깝게 축소되었지만 0이 되지는 않았습니다.")
print("-" * 50)


# ==============================================================================
# 섹션 3: alpha 값에 따른 계수 변화 시각화
# alpha 값에 따라 계수가 어떻게 변하는지 시각적으로 확인합니다.
# ==============================================================================
print("=== [Section 3] alpha 값에 따른 계수 변화 ===")
alphas = np.logspace(-2, 2, 100)
coefs = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)

plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale("log")
plt.xlabel("Alpha (Regularization Strength)")
plt.ylabel("Coefficients")
plt.title("Ridge Coefficients as a function of Alpha")
plt.legend([f"Feature {i + 1}" for i in range(5)])
plt.axis("tight")
plt.grid(True)
plt.show()

"""
실행 예시

=== [Section 1] 일반 선형 회귀 모델 ===
일반 선형 회귀 모델의 계수: [3.1631208  2.13159999 1.06827716 0.08499474 0.00957412]
MSE (오차): 3.2356
--------------------------------------------------
=== [Section 2] Ridge 모델 ===
Ridge 모델의 계수 (alpha=1.0): [3.10017058 2.09502553 1.05381601 0.0829495  0.00408486]
MSE (오차): 3.2309

- 모든 특징의 계수들이 0에 가깝게 축소되었지만 0이 되지는 않았습니다.
--------------------------------------------------
=== [Section 3] alpha 값에 따른 계수 변화 ===
"""

"""
복습

Q1. Ridge 모델에서 L2 규제가 하는 역할은 무엇이며, Lasso의 L1 규제와 어떤 차이점이 있나요?

    - Ridge의 L2 규제는 모델의 계수들의 제곱 합에 비례하여 패널티를 부과합니다.
      이 패널티는 계수 값들이 너무 커지지 않도록 억제하여 모델의 복잡도를 낮추고 과적합을 방지하는 역할을 합니다.

    - Lasso와의 차이점: Lasso의 L1 규제는 계수의 절댓값 합에 패널티를 주며, 
      이 때문에 중요도가 낮은 특징의 계수를 완전히 0으로 만듭니다. 
      반면, Ridge의 L2 규제는 계수 값을 0에 가깝게 축소시킬 뿐, 완전히 0으로 만들지는 않습니다.

Q2. Ridge는 Lasso와 달리 왜 '특징 선택(Feature Selection)' 효과가 없나요?

    - L2 규제는 손실 함수에 계수들의 제곱 합을 더하기 때문에, 
      계수 값을 0에 가깝게 만들 수는 있어도 정확히 0으로 만들기는 어렵습니다. 
      계수들이 0에 가까워지더라도 완전히 0이 되지 않기 때문에 
      Ridge는 Lasso처럼 불필요한 특징을 모델에서 제외하는 특징 선택 효과는 없습니다. 
      대신 모든 특징을 조금씩 줄여서 모델의 안정성을 높입니다.

Q3. Ridge 모델의 하이퍼파라미터인 alpha의 역할은 Lasso와 동일한가요? alpha가 매우 큰 값일 때 모델은 어떻게 될까요?

    - 네, Ridge 모델의 alpha도 Lasso와 마찬가지로 규제 강도를 조절하는 역할을 합니다. 
      alpha가 커질수록 규제가 강해집니다.

    - alpha가 매우 큰 값이라면, 규제의 패널티가 압도적으로 커지기 때문에 모델의 계수 값들은 모두 0에 매우 가깝게 수렴하게 됩니다. 
      이 경우 모델은 모든 예측을 평균값으로만 하게 되는 매우 단순한(underfitting) 모델이 됩니다.

Q4. 일반적으로 Ridge와 Lasso 중 어떤 모델을 먼저 고려해야 할까요? 두 모델을 결합한 형태의 모델도 존재하나요?

    - 고려 순서: 일반적으로 **Ridge**를 먼저 고려하는 경우가 많습니다. 
      Ridge는 모든 특징을 조금씩 축소시키기 때문에 특징들 간의 상관관계가 높을 때 더 안정적인 성능을 보입니다. 
      반면 Lasso는 불필요한 특징이 많고 그 특징을 명확하게 제거해야 할 때 유용합니다.

    - 결합 모델: 네, Lasso의 L1 규제와 Ridge의 L2 규제를 결합한 모델이 존재하며, 이를 **ElasticNet**이라고 합니다. 
      Lasso를 써야 할지 Ridge를 써야 할지 모를 때 ElasticNet을 사용하면 두 규제의 장점을 모두 활용할 수 있습니다.
      ElasticNet은 두 규제의 장점을 모두 취하여 모델의 복잡도를 줄이면서 동시에 특징 선택 효과도 얻을 수 있어, 다양한 데이터 환경에서 안정적인 성능을 기대할 수 있습니다.

    - Lasso의 fit() 함수가 사용하는 최소화 목적 함수는 다음과 같습니다:
        \[
        \text{Minimize} \quad ||y - X\beta||^2_2 + \alpha ||\beta||_1
        \]
    
    - Ridge의 fit() 함수가 사용하는 최소화 목적 함수는 다음과 같습니다:
        \[
        \text{Minimize} \quad ||y - X\beta||^2_2 + \alpha ||\beta||^2_2
        \]
    
    - ElasticNet의 fit() 함수가 사용하는 최소화 목적 함수는 다음과 같습니다:
        \[
        \text{Minimize} \quad ||y - X\beta||^2_2 + \alpha_1 ||\beta||_1 + \alpha_2 ||\beta||^2_2
        \]

Q5. 모델 시도 순서

    - 어느 모델이 더 좋은지 미리 알기 어려운 경우가 대부분이므로
      GridSearchCV 등을 활용하여 ElasticNet를 시도한 후 alpha 값을 찾아냅니다.

      alpha 값이 1에 가까우면 Lasso를 쓰는 것이 좋았다는 것을 알게 됩니다.

      alpha 값이 0에 가까우면 Ridge를 쓰는 것이 좋았다는 것을 알게 됩니다.

      alpha 값이 0과 1 사이의 값이면 ElasticNet를 쓰는 것이 좋았다는 것을 알게 됩니다.

"""
