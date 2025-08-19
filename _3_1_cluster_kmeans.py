"""
비지도 학습 (Unsupervised Learning):
    군집 (Clustering): 데이터의 구조나 패턴을 찾아 유사한 데이터끼리 그룹화하는 데 사용됩니다.
        예: sklearn.cluster.KMeans, sklearn.cluster.DBSCAN
    차원 축소 (Dimensionality Reduction): 데이터의 차원을 줄여 분석 및 시각화를 용이하게 합니다.
        예: sklearn.decomposition.PCA, sklearn.manifold.TSNE

KMenas:
    KMeans는 가장 널리 사용되는 비지도 학습(Unsupervised Learning) 알고리즘 중 하나로,
    데이터를 미리 정해진 K개의 군집(Cluster)으로 그룹화하는 데 사용됩니다.
    이 데모 코드를 통해 KMeans의 학습 과정을 단계별로 살펴보겠습니다.
"""

# 1. 필요한 라이브러리 임포트
import matplotlib.pyplot as plt  # 시각화 라이브러리
from sklearn.cluster import KMeans  # K-평균 군집 모델
from sklearn.datasets import make_blobs  # 군집용 가상 데이터 생성

# 2. 예제 데이터 생성
# '고객 세분화(Customer Segmentation)' 비즈니스 시나리오를 모사합니다.
# make_blobs 함수로 여러 개의 군집(blobs)으로 나뉜 가상의 데이터를 만듭니다.
# - n_samples=300: 300명의 고객 데이터
# - n_features=2: 2가지 고객 행동 지표 (예: '월 평균 방문 횟수', '월 평균 구매 금액')
# - centers=3: 3개의 고객 그룹 (예: '충성 고객', '일반 고객', '잠재 고객')
# - cluster_std: 군집 내 데이터의 표준편차. 값이 작을수록 군집이 명확하게 구분됩니다.
# - random_state: 재현 가능한 결과를 위한 시드값
# - y: make_blobs가 생성한 실제 군집 레이블. KMeans는 비지도 학습이므로 학습에는 사용하지 않지만,
#      나중에 모델이 얼마나 정답에 가깝게 군집을 찾았는지 평가하거나 시각화할 때 '참고용 정답지'로 사용됩니다.
X, y = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=0.60, random_state=42)

# ==============================================================================
# 섹션 2-1: 원본 데이터 시각화 (참고용)
# make_blobs가 생성한 실제 군집(y)을 기준으로 데이터를 시각화합니다.
# 이는 KMeans가 찾아야 할 '정답' 군집의 모습입니다.
# ==============================================================================
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="viridis")
plt.title("Original Data with True Labels (Ground Truth)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 3. 추정기(Estimator) 인스턴스 생성
# KMeans 모델을 생성합니다.
# n_clusters: 만들고자 하는 군집의 개수 (이것이 K값입니다).
# random_state: 재현 가능한 결과를 위한 시드값
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
# n_init은 KMeans 알고리즘을 여러 번 실행해 최적의 결과를 찾는 횟수입니다.

# 4. 모델 학습 (fit)
# model.fit(X) 메서드를 사용하여 모델을 학습시킵니다.
# 비지도 학습이므로, 정답 레이블(y) 없이 특징 데이터(X)만 사용합니다.
# 이 과정을 통해 KMeans는 데이터의 패턴을 분석해 최적의 군집 중심점(centroids)을 찾습니다.
print("K-Means 군집화 시작...")
kmeans.fit(X)
print("K-Means 군집화 완료!")

# 5. 예측 (predict) 또는 군집 레이블 할당
# predict(X) 메서드를 사용하여 각 데이터 포인트가 어떤 군집에 속하는지 예측합니다.
# fit()을 실행하면, 각 데이터 포인트에 대한 군집 할당 결과가 이미 `labels_` 속성에 저장됩니다.
# 따라서, 학습에 사용된 데이터의 군집 결과를 확인할 때는 `predict()`를 다시 호출할 필요 없이 `labels_`를 사용하는 것이 더 효율적입니다.
# y_kmeans = kmeans.predict(X)  # 이 코드도 동일한 결과를 반환합니다.
y_kmeans = kmeans.labels_

# 6. 결과 시각화
# matplotlib을 사용하여 군집 결과를 시각화합니다.
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap="viridis")

# 각 군집의 중심점을 표시합니다.
centers = kmeans.cluster_centers_
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c="red",
    s=200,
    alpha=0.7,
    marker="X",
    label="Centroids",
)
plt.title("K-Means Clustering Result")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# 7. 군집의 중심점 및 할당된 레이블 출력
print("\n각 군집의 중심점 (Centroids):")
print(centers)
print("\n첫 10개 데이터 포인트에 할당된 군집 레이블:")
print(y_kmeans[:10])

"""
K-Means 군집화 시작...
K-Means 군집화 완료!
"""

"""
복습

Q1. KMeans는 비지도 학습인데, 왜 예제 데이터를 생성할 때 정답 레이블(y)을 함께 받나요?

    KMeans는 학습 과정(`fit(X)`)에서 정답 레이블(y)을 전혀 사용하지 않기 때문에 비지도 학습이 맞습니다.
    그럼에도 y를 받는 이유는, 학습이 끝난 후 우리 모델이 얼마나 정답에 가깝게 군집을 잘 찾았는지 **평가하고 시각화**하기 위함입니다.
    즉, y는 모델의 성능을 측정하기 위한 **'참고용 정답지'** 역할을 합니다.
    실제 현업 데이터에는 이러한 정답 레이블이 없는 경우가 대부분입니다.

Q2. n_clusters 매개변수의 역할은 무엇이며, 이 값을 잘못 설정하면 어떤 문제가 발생할 수 있나요?

    **n_clusters**는 KMeans 알고리즘이 데이터를 나눌 **군집의 개수(K)**를 지정하는 매개변수입니다.
    이 값을 잘못 설정하면 데이터의 실제 구조와 다른 결과가 나올 수 있습니다. 
    예를 들어, 실제 데이터가 4개의 군집으로 구성되어 있는데 n_clusters=2로 설정하면, 
    실제로는 다른 군집인 데이터들이 하나의 군집으로 묶여 버리는 문제가 발생합니다.

Q3. 코드에서 kmeans.fit(X)를 실행한 후, kmeans.predict(X)를 다시 호출하는 대신 kmeans.labels_ 속성을 사용하는 이유는 무엇인가요?

    KMeans 모델은 fit() 메서드를 호출하는 과정에서 이미 각 데이터 포인트가 
    어떤 군집에 속하는지 계산하고, 그 결과를 `labels_` 속성에 저장합니다.
    따라서 `predict()` 메서드를 다시 호출하는 것은 이미 계산된 결과를 다시 찾는 과정과 유사합니다. 
    `labels_` 속성은 학습 완료 후 바로 접근할 수 있는 결과이므로, 코드의 효율성과 가독성을 위해 직접 사용하는 경우가 많습니다.

Q4. 이 예제에서는 왜 train/test 데이터를 분리하지 않고 전체 데이터(X)로 학습하고 예측하나요?

    아주 중요한 포인트입니다. 이는 KMeans가 **비지도 학습**이기 때문입니다.

    - **지도 학습 (분류/회귀)**: 모델의 **일반화 성능**을 평가하는 것이 중요합니다. 
      즉, 모델이 처음 보는 데이터(Test set)를 얼마나 잘 예측하는지 확인해야 하므로, 반드시 데이터를 분리해야 합니다.

    - **비지도 학습 (군집)**: 주된 목적은 주어진 데이터셋 전체의 **내재된 구조(군집)를 발견**하는 것입니다. 
      '정답'이 없기 때문에 일반화 성능을 평가하는 개념이 다릅니다. 
      따라서, 가지고 있는 데이터 전체를 사용하여 어떤 군집들이 존재하는지 파악하는 것이 일반적인 접근 방식입니다.

    물론, 군집화 결과를 평가하는 다른 기법(예: 실루엣 점수)도 있지만, 기본적인 군집화 작업에서는 전체 데이터를 사용하는 것이 표준입니다.

Q5. K-Means 알고리즘의 작동 원리를 간단하게 설명해 주세요.

    K-Means는 다음과 같은 과정을 반복하며 작동합니다.

    1. 초기화: 사용자가 지정한 군집의 개수(K)만큼 **무작위로 군집의 중심점(Centroids)**을 설정합니다.
    2. 할당: 모든 데이터 포인트를 가장 가까운 군집 중심점에 할당합니다.
    3. 갱신: 각 군집에 할당된 데이터 포인트들의 평균 위치를 계산하여 새로운 군집 중심점으로 갱신합니다.

    이 과정은 군집 중심점의 위치가 더 이상 변하지 않을 때까지 반복됩니다.

Q6. 실수로 초기 중심점이 매우 가까운 곳이 선택되어 로컬 미니멈에 빠질 우려가 있는데 그걸 피하기 위한 방법은?

    KMeans 알고리즘은 초기 중심점의 선택에 민감하여, 잘못된 초기값이 로컬 미니멈에 빠질 수 있습니다.

    **KMeans++ 초기화**: KMeans++ 알고리즘을 사용하면 초기 중심점을 더 스마트하게 선택하여 로컬 미니멈에 빠질 확률을 줄일 수 있습니다.

    이는 sklearn의 KMeans에서 기본적으로 적용되는 방법입니다.    
"""
