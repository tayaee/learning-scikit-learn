"""
지도 학습 (Supervised Learning):
    분류 (Classification): 데이터를 미리 정의된 클래스 또는 범주로 분류하는 데 사용됩니다.
        예: sklearn.svm.SVC, sklearn.ensemble.RandomForestClassifier
    회귀 (Regression): 연속적인 숫자 값을 예측하는 데 사용됩니다.
        예: sklearn.linear_model.LinearRegression, sklearn.ensemble.RandomForestRegressor

LR:
    LinearRegression은 가장 기본적인 선형 회귀 모델로,
    데이터 간의 선형 관계를 모델링하여 연속적인 값을 예측하는 데 사용됩니다.
    이 데모 코드를 통해 LinearRegression의 학습 과정을 단계별로 살펴보겠습니다.
"""

# 1. 필요한 라이브러리 임포트
from sklearn.datasets import make_regression  # 회귀용 가상 데이터 생성
from sklearn.linear_model import LinearRegression  # 선형 회귀 모델
from sklearn.metrics import mean_squared_error, r2_score  # 모델 평가 지표
from sklearn.model_selection import train_test_split  # 데이터 분할

# 2. 예제 데이터 생성
# '일일 광고비'를 바탕으로 '일일 매출액'을 예측하는 비즈니스 시나리오를 모사합니다.
# - n_samples=200: 200일간의 데이터
# - n_features=1: 1개의 특징 (독립 변수), 즉 '일일 광고비'
# - y (타겟 변수): '일일 매출액'
# - noise=20: 광고비 외에 매출에 영향을 주는 예측 불가능한 요인 (날씨, 경쟁사 이벤트 등)
# - random_state=42: 재현 가능한 결과를 위한 시드값
X, y = make_regression(n_samples=200, n_features=1, noise=20, random_state=42)

# 3. 데이터 분할
# 전체 데이터를 학습(train) 세트와 테스트(test) 세트로 나눕니다.
# 학습 세트(80%)는 모델 훈련에 사용하고, 테스트 세트(20%)는 모델 성능 평가에 사용합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 추정기(Estimator) 인스턴스 생성
# LinearRegression 모델을 생성합니다.
model = LinearRegression()

# 5. 모델 학습 (fit)
# model.fit(X, y) 메서드를 사용하여 학습 데이터에 모델을 맞춥니다.
# 이 과정을 통해 모델은 데이터에 가장 잘 맞는 회귀 직선의 기울기(coefficient)와 절편(intercept)을 찾습니다.
print("모델 학습 시작...")
model.fit(X_train, y_train)
print("모델 학습 완료!")

# 6. 예측 (predict)
# model.predict(X) 메서드를 사용하여 학습된 모델로 테스트 데이터에 대한 예측을 수행합니다.
# 예측 결과는 연속적인 숫자 값(회귀 값)이 됩니다.
y_pred = model.predict(X_test)

# 7. 모델 평가
# 실제 값(y_test)과 예측 값(y_pred)을 비교하여 모델의 성능을 평가합니다.

# 평균 제곱 오차 (Mean Squared Error, MSE): 예측 값과 실제 값의 차이(오차)를 제곱한 값들의 평균
mse = mean_squared_error(y_test, y_pred)
print(f"\n평균 제곱 오차 (MSE): {mse:.2f}")

# 결정 계수 (R-squared, R2 Score): 모델이 분산을 얼마나 잘 설명하는지 나타내는 지표. 1에 가까울수록 좋습니다.
r2 = r2_score(y_test, y_pred)
print(f"결정 계수 (R2 Score): {r2:.2f}")

# 모델의 계수와 절편 출력
print("\n모델의 계수 (coefficient, 기울기):", model.coef_[0])
print("모델의 절편 (intercept):", model.intercept_)

"""
모델 학습 시작...
모델 학습 완료!

평균 제곱 오차 (MSE): 437.55
결정 계수 (R2 Score): 0.94

모델의 계수 (coefficient, 기울기): 86.5115419768739
모델의 절편 (intercept): 2.4461021846792352 
"""

"""
복습

Q1. LinearRegression 모델은 어떤 유형의 머신러닝 문제에 주로 사용되나요? SVC나 RandomForestClassifier와는 어떤 차이가 있나요?

    - LinearRegression은 회귀(Regression) 문제, 즉 연속적인 숫자 값을 예측하는 데 사용되는 지도 학습 모델입니다.

    - SVC나 RandomForestClassifier는 분류(Classification) 문제, 즉 데이터를 미리 정해진 범주로 나누는 데 사용됩니다. 
      회귀는 연속적인 값을, 분류는 이산적인 클래스 레이블을 예측한다는 점에서 가장 큰 차이가 있습니다.

Q2. 회귀 모델에서 fit() 메서드를 호출할 때, 모델이 학습하는 주요 파라미터는 무엇인가요?

    - LinearRegression 모델은 fit() 메서드를 통해 입력 데이터(X_train)와 정답(y_train) 간의 관계를 
      가장 잘 나타내는 **기울기(coefficient)**와 **절편(intercept)**을 학습합니다. 
      이 파라미터들이 바로 모델을 정의하는 핵심 요소입니다.

Q3. mean_squared_error와 r2_score는 각각 모델의 어떤 측면을 평가하는 데 사용되나요?

    - mean_squared_error (MSE): 예측값과 실제값의 차이(오차)를 제곱해 평균을 낸 값으로, 
      예측의 정확도를 측정하는 데 사용됩니다. 값이 0에 가까울수록 모델의 예측이 정확하다는 것을 의미합니다.

    - r2_score (결정 계수): 모델이 데이터의 분산을 얼마나 잘 설명하는지를 나타내는 지표입니다. 
      0에서 1 사이의 값을 가지며, 1에 가까울수록 모델이 데이터를 잘 설명한다고 판단할 수 있습니다.

Q4. noise 매개변수가 회귀 모델의 학습 및 성능 평가에 어떤 영향을 미칠까요?

    - noise 매개변수는 데이터에 무작위로 추가되는 잡음을 의미합니다.

    - 이 값이 클수록 데이터가 회귀 직선에서 더 많이 흩어지게 되므로, 모델이 데이터의 패턴을 학습하기가 더 어려워집니다. 
      이는 모델의 성능을 떨어뜨리는 요인이 됩니다.
    
    - 따라서, noise 값이 클수록 mean_squared_error는 증가하고, r2_score는 감소하는 경향을 보입니다.
"""
