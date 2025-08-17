"""
전처리 (Preprocessing):
    데이터를 모델이 학습하기에 적합한 형태로 변환합니다. 결측값 처리, 데이터 스케일링, 특징 추출 등이 포함됩니다.
        예: sklearn.preprocessing.StandardScaler, sklearn.impute.SimpleImputer

StandardScaler는 데이터 전처리 단계에서 가장 흔하게 사용되는 변환기(Transformer)입니다.
각 특징(feature)의 평균을 0, 표준편차를 1로 조정하여 데이터의 스케일을 통일하는 역할을 합니다.
이는 여러 특징의 스케일이 다른 데이터셋에서 모델이 특정 특징에 편향되지 않도록 하는 데 중요합니다.
"""

# 1. 필요한 라이브러리 임포트
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression  # 모델 학습 비교용
from sklearn.model_selection import train_test_split  # 데이터 분할
from sklearn.preprocessing import StandardScaler  # StandardScaler 변환기

# 2. 예제 데이터 생성
# 실제 데이터에서 흔히 볼 수 있는 '특징 간 스케일 차이'를 인위적으로 만듭니다.
# 예를 들어, 아파트 가격을 예측하는 모델을 만든다고 가정해 봅시다.
# - 100개의 데이터 샘플 (100채의 아파트)
# - 2개의 특징 (방의 개수, 집의 면적)
# X[:, 0]: '방의 개수' (0~10 사이의 작은 스케일)
# X[:, 1]: '집의 면적(m²)' (0~1000 사이의 큰 스케일)
X = np.random.rand(100, 2) * np.array([10, 1000])

# 타겟(y) 데이터는 단순히 X의 합에 약간의 노이즈를 더한 것으로 설정
y = X[:, 0] + X[:, 1] + np.random.randn(100) * 10

# 3. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---
# 4. Standard Scaler 적용
# Standard Scaler 추정기(변환기) 인스턴스 생성
scaler = StandardScaler()

# fit_transform: 학습 데이터에 대한 평균과 표준편차를 계산(fit)하고 데이터를 표준화(transform)합니다.
# 이 과정이 가장 중요한 단계입니다.
X_train_scaled = scaler.fit_transform(X_train)

# transform: 테스트 데이터는 학습 데이터의 통계량(평균, 표준편차)을 이용해 표준화만 합니다.
# 새로운 데이터를 만난 상황을 시뮬레이션하며, 데이터 누수(data leakage)를 방지합니다.
X_test_scaled = scaler.transform(X_test)

# ---
# 5. 스케일링 전후 데이터 비교
print("스케일링 전 학습 데이터의 평균:")
print(f"특징 1: {X_train[:, 0].mean():.2f}, 특징 2: {X_train[:, 1].mean():.2f}")
print("\n스케일링 후 학습 데이터의 평균:")
print(f"특징 1: {X_train_scaled[:, 0].mean():.2f}, 특징 2: {X_train_scaled[:, 1].mean():.2f}")

print("\n스케일링 전 학습 데이터의 표준편차:")
print(f"특징 1: {X_train[:, 0].std():.2f}, 특징 2: {X_train[:, 1].std():.2f}")
print("\n스케일링 후 학습 데이터의 표준편차:")
print(f"특징 1: {X_train_scaled[:, 0].std():.2f}, 특징 2: {X_train_scaled[:, 1].std():.2f}")

# 6. 시각화를 통한 전후 비교
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X_train[:, 0], X_train[:, 1])
axes[0].set_title("Original Data")
axes[0].set_xlabel("Feature 1 (Small Scale)")
axes[0].set_ylabel("Feature 2 (Large Scale)")
axes[0].axis("equal")

axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1])
axes[1].set_title("StandardScaled Data")
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")
axes[1].axis("equal")

plt.tight_layout()
plt.show()

# 7. (보너스) 모델 학습에 미치는 영향 비교
model_raw = LinearRegression().fit(X_train, y_train)
model_scaled = LinearRegression().fit(X_train_scaled, y_train)

# 회귀 모델의 계수(기울기) 비교
print("\n스케일링 전 모델의 계수:", model_raw.coef_)
print("스케일링 후 모델의 계수:", model_scaled.coef_)

"""
실행 결과

스케일링 전 학습 데이터의 평균:
특징 1: 5.13, 특징 2: 506.97

스케일링 후 학습 데이터의 평균:
특징 1: -0.00, 특징 2: 0.00

스케일링 전 학습 데이터의 표준편차:
특징 1: 2.78, 특징 2: 277.18

스케일링 후 학습 데이터의 표준편차:
특징 1: 1.00, 특징 2: 1.00

스케일링 전 모델의 계수: [1.36303872 1.00043582]
스케일링 후 모델의 계수: [3.79430107 277.30533981]

# 이때 모델이 학습한 계수를 보면, '표준화된 집의 면적'의 계수(277.30)가 '표준화된 방의 개수'의 계수(3.79)보다 압도적으로 큽니다.
# 이는 "집의 면적"이 "방의 개수"보다 집 가격을 예측하는 데 있어 훨씬 더 중요한 특징이라는 것을 직관적으로 보여줍니다.
# 스케일링을 하지 않았다면 더 중요한 집의 면적의 중요성이 왜곡되어, 모델이 잘못된 예측을 할 수 있습니다.
# 따라서, StandardScaler와 같은 스케일링 기법은 모델의 성능을 향상시키는 데 중요한 역할을 합니다.
"""

"""
코드 복기

Q1. StandardScaler의 주요 역할은 무엇이며, 왜 머신러닝 모델을 학습하기 전에 데이터를 스케일링하는 것이 중요한가요?

    - StandardScaler는 각 특징의 평균을 0, 표준편차를 1로 조정하여 데이터의 스케일을 통일하는 역할을 합니다.

    - 스케일링이 중요한 이유는, 많은 머신러닝 알고리즘(특히 경사하강법 기반 모델, SVM, PCA 등)이 특징 간의 거리를 기반으로 작동하기 때문입니다. 
      스케일이 크게 다른 특징이 있으면, 모델은 스케일이 큰 특징에 더 민감하게 반응하여 학습이 제대로 이루어지지 않거나 성능이 저하될 수 있습니다.

Q2. StandardScaler에서 fit_transform()과 transform() 메서드를 각각 학습 데이터와 테스트 데이터에 사용하는 이유는 무엇인가요?

    - 학습 데이터(X_train)에 fit_transform()을 사용하는 이유: 
      학습 데이터의 **평균과 표준편차를 계산(fit)**하고, 그 통계량을 사용해 데이터를 **표준화(transform)**하기 위함입니다. 
      모델은 이 통계량으로 학습하게 됩니다.
    
    - 테스트 데이터(X_test)에 transform()만 사용하는 이유: 
      테스트 데이터는 미지의 데이터를 시뮬레이션해야 하므로, 
      학습 데이터에서 얻은 통계량(평균, 표준편차)을 그대로 적용하여 표준화해야 합니다. 
      테스트 데이터의 통계량을 따로 계산(fit)하면, 
      학습 과정에 테스트 데이터의 정보가 유출되는 **데이터 누수(Data Leakage)**가 발생하여 
      모델 성능이 과대평가(실제 성능보다 더 좋게 측정)될 수 있습니다.

Q3. 코드에서 스케일링 전후의 회귀 모델 계수(model.coef_)가 다르게 나오는 것을 볼 수 있습니다. 이것이 의미하는 바는 무엇인가요?

    - 이는 모델이 학습한 데이터의 스케일에 따라 coef_의 상대적인 크기와 값이 달라진다는 것을 의미합니다.

    - 스케일링 전에는 스케일이 큰 특징(Feature 2)의 계수 값도 상대적으로 작아질 수 있습니다. 
      반면, 스케일링 후에는 모든 특징이 동일한 스케일을 가지므로, 
      각 특징이 타겟 값에 미치는 영향력의 상대적인 크기를 계수 값으로 더 직관적으로 비교할 수 있게 됩니다.

Q4. StandardScaler와 같은 전처리 변환기도 fit()과 transform() 메서드를 가집니다. 
    이를 추정기(Estimator) API 관점에서 어떻게 설명할 수 있나요?

    - scikit-learn의 모든 객체는 **추정기(Estimator)**라는 일관된 API를 따릅니다. 
      이 추정기 API의 핵심은 fit(), predict() 또는 transform() 같은 메서드입니다.
    
    - StandardScaler는 데이터를 학습하는 대신, 데이터의 통계량(평균, 표준편차)을 **학습(fit)**하고, 
      이를 이용해 데이터를 **변환(transform)**한다는 점에서 추정기 API를 충실히 구현하고 있습니다. 
      predict()가 없는 대신 transform()을 사용하여 모델을 학습시키기 위한 전처리 단계의 일관성을 유지하는 것입니다.
"""
