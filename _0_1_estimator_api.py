"""
1. 추정기 (Estimator) API
    scikit-learn의 모든 머신러닝 알고리즘은 '추정기(Estimator)'라는 일관된 객체를 통해 구현됩니다.
    추정기는 데이터로부터 학습하는 모든 객체를 의미하며, 분류기, 회귀 모델, 변환기 등 다양한 종류가 있습니다.
    추정기 API의 핵심 메서드는 다음과 같습니다.
        fit(X, y): 모델을 데이터에 맞추는(학습시키는) 메서드입니다. X는 입력 데이터(특징), y는 타겟 레이블(정답)을 나타냅니다.
        predict(X): 학습된 모델을 사용하여 새로운 데이터 X에 대한 예측을 수행하는 메서드입니다.
        transform(X): 데이터 전처리 단계에서 사용되는 메서드입니다. 입력 데이터를 변환하여 출력합니다.
        fit_transform(X, y): fit과 transform을 순서대로 수행하는 메서드로, 효율성을 위해 자주 사용됩니다.
"""

# 1. 필요한 라이브러리 임포트
import numpy as np
from sklearn.linear_model import LinearRegression  # 선형 회귀 모델 (추정기)
from sklearn.preprocessing import StandardScaler  # 데이터 스케일링 (변환기)
from sklearn.model_selection import train_test_split  # 데이터 분할

# 2. 예제 데이터 생성
# X: 독립 변수(집 크기), y: 종속 변수(집 가격)
# reshape(-1, 1)을 사용하여 1차원 배열을 2차원 배열로 변환합니다.
# -1은 배열의 길이를 기반으로 차원 크기를 자동으로 추론하도록 합니다.
# scikit-learn은 2차원 배열 형태의 입력 데이터를 기대합니다.
X = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50]).reshape(-1, 1)
y = np.array([12, 18, 22, 28, 33, 38, 45, 52, 58])

# 3. 데이터 분할
# 학습(train) 세트와 테스트(test) 세트로 데이터를 나눕니다.
# 학습 세트는 모델 훈련에 사용되고, 테스트 세트는 모델 성능 평가에 사용됩니다.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---
# 4. 추정기 API 사용 예시 (변환기 - StandardScaler)
# StandardScaler는 데이터를 평균 0, 표준편차 1로 표준화하는 역할을 하는 변환기입니다.
scaler = StandardScaler()

# fit_transform(): 학습 데이터의 평균과 표준편차를 계산(fit)하고 데이터를 표준화(transform)합니다.
# 학습 데이터에 대한 정보(평균, 표준편차)는 이후 테스트 데이터 변환에도 사용됩니다.
# 평균을 0, 표준편차를 1로 맞춥니다. 정규 분포를 가정하고 있는 것은 아닙니다.
X_train_scaled = scaler.fit_transform(X_train)

# transform(): 테스트 데이터는 학습 데이터의 통계량(평균, 표준편차)을 이용해 표준화만 합니다.
# 테스트 데이터로 fit을 하면 데이터 누수(data leakage)가 발생할 수 있습니다.
X_test_scaled = scaler.transform(X_test)

# ---
# 5. 추정기 API 사용 예시 (회귀 모델 - LinearRegression)
# LinearRegression은 선형 회귀 모델을 구현한 추정기입니다.
model = LinearRegression()

# fit(X, y): 학습 데이터(X_train_scaled, y_train)를 사용하여 모델을 학습시킵니다.
# 이 과정을 통해 모델은 최적의 회귀 계수를 찾습니다.
model.fit(X_train_scaled, y_train)

# ---
# 6. 예측 (predict)
# predict(X): 학습된 모델을 사용하여 테스트 데이터에 대한 예측을 수행합니다.
# 예측 시에는 학습에 사용된 것과 동일하게 스케일링된 데이터를 사용해야 합니다.
predictions = model.predict(X_test_scaled)

# 7. 결과 출력
print("원래 테스트 데이터:\n", X_test)
print("스케일링된 테스트 데이터:\n", X_test_scaled)
print("실제 집 가격:\n", y_test)
print("예측된 집 가격:\n", predictions)

# 모델의 계수와 절편 출력
print("\n모델의 계수 (coefficient):", model.coef_)
print("모델의 절편 (intercept):", model.intercept_)

"""
실행 예:

원래 테스트 데이터:
 [[45]
 [15]
 [35]]
스케일링된 테스트 데이터:
 [[ 1.21386547]
 [-1.08609016]
 [ 0.4472136 ]]
실제 집 가격:
 [52 18 38]
예측된 집 가격:
 [51.2244898  16.69387755 39.71428571]  

모델의 계수 (coefficient): [15.01359928]
모델의 절편 (intercept): 33.0
"""

"""
코드 복기

Q1. LinearRegression과 StandardScaler는 각각 어떤 API 역할을 수행하나요?
    
    - **LinearRegression**은 지도 학습의 회귀 모델을 구현한 **추정기(Estimator)**입니다. 
      데이터의 패턴을 학습해 연속적인 값을 예측하는 역할을 합니다.

    - **StandardScaler**는 데이터의 특징을 표준화하는 변환기(Transformer) 역할을 합니다. 
      이는 모델이 데이터를 더 효율적으로 학습할 수 있도록 전처리하는 과정에서 사용됩니다.

Q2. StandardScaler 인스턴스인 scaler에 대해 fit_transform()과 transform() 메서드를 
    각각 X_train과 X_test에 사용하는 이유는 무엇인가요? X_test에도 fit_transform()을 사용하면 안 되는 이유는요?
    
    - X_train에 fit_transform()을 사용하는 이유: 
      학습 데이터인 X_train에 대한 **평균과 표준편차를 계산(fit)**하고, 
      그 통계량을 사용해 데이터를 **표준화(transform)**하기 위함입니다. 
      모델은 이 통계량으로 학습하게 됩니다.
    
    - X_test에 transform()만 사용하는 이유: 
      X_test 데이터는 학습에 관여하지 않는 미지의 데이터로 간주해야 합니다. 
      fit_transform()을 사용해 X_test의 통계량을 계산하게 되면, 
      학습 데이터에 대한 정보가 테스트 데이터에 **유출(Data Leakage)**되어 
      모델의 일반화 성능을 정확히 평가할 수 없게 됩니다.
      따라서 학습 데이터에서 얻은 통계량을 그대로 적용해 표준화만 해야 합니다.

Q3. predict() 메서드는 어떤 데이터를 인자로 받아야 하나요? 왜 원본 X_test가 아닌 X_test_scaled를 사용해야 하나요?
    
    - predict() 메서드는 모델 학습에 사용된 데이터와 동일한 형태로 전처리된 데이터를 인자로 받아야 합니다.
    
    - 이 코드에서는 StandardScaler로 표준화된 X_train_scaled 데이터를 이용해 LinearRegression 모델을 학습시켰습니다. 
      따라서 예측 시에도 동일한 통계량으로 표준화된 X_test_scaled를 사용해야 모델이 올바른 예측을 할 수 있습니다.

Q4. 머신러닝 모델의 학습은 어느 단계에서 이루어지나요? 그리고 이때 fit() 메서드는 어떤 역할을 합니까?

    - 머신러닝 모델의 학습은 코드의 5번 단계에서 model.fit(X_train_scaled, y_train)이 호출될 때 이루어집니다.

    - 이때 fit() 메서드는 주어진 학습 데이터(X_train_scaled)와 정답(y_train)을 바탕으로 
      모델이 최적의 **가중치(계수)**와 **편향(절편)**을 찾아내는 역할을 합니다. 
      이 과정을 통해 모델은 데이터의 경향성을 파악하고 새로운 데이터에 대한 예측을 수행할 수 있는 상태가 됩니다.
"""
