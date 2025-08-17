"""
특징(Feature) 전처리 및 엔지니어링
    모델 학습에 적합한 형태로 데이터를 가공하는 다양한 기술들입니다.
        sklearn.preprocessing.OneHotEncoder: 범주형 데이터를 원-핫 인코딩하여 머신러닝 모델이 이해할 수 있는 형태로 변환합니다. (예: '서울', '부산' -> [1, 0], [0, 1])
        sklearn.preprocessing.MinMaxScaler: 데이터를 0과 1 사이의 값으로 스케일링합니다. StandardScaler와는 또 다른 중요한 스케일링 방법입니다.
        sklearn.feature_extraction.text: 텍스트 데이터에서 특징을 추출하는 기능입니다. CountVectorizer나 TfidfVectorizer가 대표적입니다.

MinMaxScaler:
    MinMaxScaler는 데이터 스케일링을 위한 또 다른 중요한 전처리 도구입니다.
    이 변환기는 모든 특징(feature)의 값을 0과 1 사이의 특정 범위로 변환합니다.
    StandardScaler와 달리, MinMaxScaler는 데이터의 분포를 변경하지 않고 단순히 값의 범위를 재조정하여 데이터의 스케일을 통일합니다.
"""

# 1. 필요한 라이브러리 임포트
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 2. 예제 데이터 생성
# np.random.normal() 함수를 사용하여 두 특징의 스케일이 다른 데이터셋을 생성합니다.
# - loc: 정규분포의 평균 (첫 번째 특징: 50, 두 번째 특징: 5)
# - scale: 정규분포의 표준편차 (첫 번째 특징: 10, 두 번째 특징: 2)
# - size: 생성할 데이터의 형태 (100개의 샘플, 2개의 특징)
mean = [50, 5]
std = [10, 2]
X = np.random.normal(loc=mean, scale=std, size=(100, 2))

# 3. 데이터 분할
# 학습용과 테스트용 데이터를 분할합니다.
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

# ---
# 4. MinMax Scaler 적용
# MinMax Scaler 추정기(변환기) 인스턴스 생성
scaler = MinMaxScaler()

# fit_transform: 학습 데이터(X_train)의 각 특징별 최솟값과 최댓값을 계산(fit)하고,
#              이 값을 기준으로 데이터를 0과 1 사이로 스케일링(transform)합니다.
#              결과적으로 X_train_scaled의 각 특징별 최솟값은 0, 최댓값은 1이 됩니다.
X_train_scaled = scaler.fit_transform(X_train)

# transform: 테스트 데이터는 학습 데이터의 최댓값과 최솟값을 이용해 스케일링만 합니다.
X_test_scaled = scaler.transform(X_test)

# ---
# 5. 스케일링 전후 데이터 비교
print("스케일링 전 학습 데이터의 최솟값:")
print(f"특징 1: {X_train[:, 0].min():.2f}, 특징 2: {X_train[:, 1].min():.2f}")
print("\n스케일링 후 학습 데이터의 최솟값:")
print(f"특징 1: {X_train_scaled[:, 0].min():.2f}, 특징 2: {X_train_scaled[:, 1].min():.2f}")

print("\n스케일링 전 학습 데이터의 최댓값:")
print(f"특징 1: {X_train[:, 0].max():.2f}, 특징 2: {X_train[:, 1].max():.2f}")
print("\n스케일링 후 학습 데이터의 최댓값:")
print(f"특징 1: {X_train_scaled[:, 0].max():.2f}, 특징 2: {X_train_scaled[:, 1].max():.2f}")


# 6. 시각화를 통한 전후 비교
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X_train[:, 0], X_train[:, 1])
axes[0].set_title("Original Data")
axes[0].set_xlabel("Feature 1 (Large Scale)")
axes[0].set_ylabel("Feature 2 (Small Scale)")
axes[0].set_xlim(-10, 100)
axes[0].set_ylim(-10, 10)
axes[0].grid(True)

axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1])
axes[1].set_title("MinMaxScaler Data")
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")
axes[1].set_xlim(-0.1, 1.1)
axes[1].set_ylim(-0.1, 1.1)
axes[1].grid(True)

plt.tight_layout()
plt.show()

"""
실행 예:

스케일링 전 학습 데이터의 최솟값:
특징 1: 29.21, 특징 2: -0.66

스케일링 후 학습 데이터의 최솟값:
특징 1: 0.00, 특징 2: 0.00

스케일링 전 학습 데이터의 최댓값:
특징 1: 76.78, 특징 2: 10.91

스케일링 후 학습 데이터의 최댓값:
특징 1: 1.00, 특징 2: 1.00
"""

"""
복습

Q1. MinMaxScaler의 주요 역할은 무엇이며, StandardScaler와 비교했을 때 어떤 차이점이 있나요?

    - MinMaxScaler는 모든 특징의 값을 0과 1 사이의 특정 범위로 변환하는 역할을 합니다.

    - StandardScaler와의 차이점:
        StandardScaler: 각 특징의 평균을 0, 표준편차를 1로 맞춥니다. 데이터의 분포 형태는 유지한 채 스케일만 조정합니다.
        MinMaxScaler: 각 특징의 최솟값을 0, 최댓값을 1로 맞춰 값의 범위를 재조정합니다. 데이터의 분포 형태는 유지됩니다.

Q2. MinMaxScaler를 사용하면 왜 데이터의 분포가 변경되지 않고, 값의 범위만 재조정되는지 설명해 주세요.

    - MinMaxScaler의 변환 공식은 (x - min) / (max - min)입니다. 
      이 공식은 데이터를 선형적으로 변환하는 방식입니다. 
      즉, 원래 데이터가 가지고 있던 값들 간의 상대적인 간격이나 분포 형태(예: 정규분포, 왜곡된 분포 등)는 그대로 유지하면서, 
      전체적인 값의 범위만 0과 1 사이로 압축합니다.

Q3. MinMaxScaler는 데이터에 이상치(Outlier)가 있을 때 어떤 문제가 발생할 수 있나요?

    MinMaxScaler는 최댓값과 최솟값을 기준으로 스케일링을 수행하기 때문에 이상치에 매우 민감합니다.
    만약 데이터에 극단적으로 큰 이상치가 있다면, 이 이상치가 최댓값(max)으로 설정됩니다. 
    그러면 나머지 대부분의 데이터들은 0에 가까운 매우 작은 값으로 변환되어 특징들이 제대로 구분되지 않게 됩니다.

Q4. MinMaxScaler와 StandardScaler 중 어떤 스케일러를 선택해야 하는지 결정하는 기준은 무엇인가요?

    StandardScaler:
        이상치에 덜 민감하므로, 데이터에 이상치가 존재할 때 더 안정적인 성능을 보입니다.
        경사하강법을 사용하는 모델(예: 로지스틱 회귀, SVM, 신경망)이나 주성분 분석(PCA)과 같이 
        가우시안 분포를 가정하는 알고리즘에 일반적으로 더 적합합니다.

    MinMaxScaler:
        데이터의 정확한 최댓값과 최솟값이 중요하거나, 모든 특징을 0과 1 사이의 동일한 범위로 강제해야 할 때 사용됩니다.
        이미지 처리와 같이 데이터가 이미 특정 범위(0-255)로 정해져 있을 때 유용하게 사용될 수 있습니다.

    일반적으로는 StandardScaler를 먼저 시도하고, 성능이 개선되지 않거나 모델의 특성에 따라 MinMaxScaler를 고려하는 것이 좋습니다.
"""
