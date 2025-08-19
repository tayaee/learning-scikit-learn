"""
비지도 학습 (Unsupervised Learning):
    군집 (Clustering): 데이터의 구조나 패턴을 찾아 유사한 데이터끼리 그룹화하는 데 사용됩니다.
        예: sklearn.cluster.KMeans, sklearn.cluster.DBSCAN
    차원 축소 (Dimensionality Reduction): 데이터의 차원을 줄여 분석 및 시각화를 용이하게 합니다.
        예: sklearn.decomposition.PCA, sklearn.manifold.TSNE

DBSCAN:
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise)은 밀도 기반의 군집 알고리즘으로,
    미리 군집의 개수를 지정할 필요 없이 데이터의 밀집도를 기준으로 군집을 찾습니다.
    또한, 군집에 속하지 않는 이상치(Outlier)를 자동으로 감지할 수 있는 특징이 있습니다.
"""

# 1. 필요한 라이브러리 임포트
import matplotlib.pyplot as plt  # 시각화 라이브러리
import numpy as np
from sklearn.cluster import DBSCAN  # DBSCAN 군집 모델
from sklearn.datasets import make_moons  # 군집용 가상 데이터 생성

# 2. 예제 데이터 생성
# make_moons 함수로 초승달 모양의 데이터를 생성합니다.
# 이 데이터는 KMeans와 같은 거리 기반 알고리즘으로는 잘 분리되지 않는 형태입니다.
# n_samples: 생성할 데이터 샘플의 수
# noise: 데이터에 추가될 노이즈(잡음)
# random_state: 재현 가능한 결과를 위한 시드값
X, y = make_moons(n_samples=200, noise=0.05, random_state=42)


# 3. 추정기(Estimator) 인스턴스 생성
# DBSCAN 모델을 생성합니다.
# eps (epsilon): 이웃으로 간주될 최대 거리. 이 값보다 가까운 데이터 포인트는 같은 군집으로 간주될 가능성이 있습니다.
# min_samples: 군집을 형성하기 위한 최소 데이터 포인트 수.
dbscan = DBSCAN(eps=0.3, min_samples=5)

# 4. 모델 학습 (fit)
# model.fit(X) 메서드를 사용하여 모델을 학습시킵니다.
# 비지도 학습이므로, 정답 레이블(y) 없이 특징 데이터(X)만 사용합니다.
# 이 과정을 통해 DBSCAN은 데이터의 밀집된 영역을 찾아 군집을 형성합니다.
print("Starting DBSCAN clustering...")
dbscan.fit(X)
print("DBSCAN clustering complete!")

# 5. 군집 레이블 할당
# fit() 메서드 실행 후, 각 데이터 포인트에 할당된 군집 레이블은 labels_ 속성에 저장됩니다.
# -1은 노이즈(이상치)를 의미하며, 0부터 시작하는 정수는 각 군집을 나타냅니다.
labels = dbscan.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

# 6. 결과 시각화
plt.figure(figsize=(8, 6))
# 각 군집을 색상으로 구분하여 시각화합니다.
# 이상치(-1)는 검은색으로 표시합니다.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # 검은색으로 이상치 표시
        col = [0, 0, 0, 1]

    class_member_mask = labels == k
    xy = X[class_member_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title(f"DBSCAN Clustering Result\nEstimated number of clusters: {n_clusters_}, Number of noise points: {n_noise_}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 7. 군집 정보 출력
print(f"\n감지된 군집의 수 (노이즈 제외): {n_clusters_}")
print(f"노이즈로 분류된 데이터 포인트의 수: {n_noise_}")

"""
실행 예

Starting DBSCAN clustering...
DBSCAN clustering complete!

감지된 군집의 수 (노이즈 제외): 2
노이즈로 분류된 데이터 포인트의 수: 0
"""

"""
복습

Q1. DBSCAN은 KMeans와 달리 군집의 개수를 미리 지정하지 않아도 됩니다. 이는 DBSCAN의 어떤 특징 때문인가요?

    DBSCAN은 밀도 기반으로 군집을 형성하기 때문입니다. 
    특정 영역에 데이터 포인트들이 충분히 밀집되어 있으면 이를 하나의 군집으로 간주합니다. 
    따라서 사용자가 미리 군집의 개수(K)를 정해줄 필요가 없으며, 
    알고리즘이 데이터의 밀집도를 스스로 분석하여 군집의 수를 결정합니다.

Q2. DBSCAN의 주요 매개변수인 eps와 min_samples는 각각 어떤 역할을 하며, 이들을 조정하면 결과에 어떤 영향을 미치나요?

    eps (epsilon): 한 데이터 포인트로부터 이웃으로 간주될 최대 거리를 지정합니다. 
    eps를 크게 설정하면 더 많은 데이터 포인트가 이웃으로 포함되어 더 큰 군집이 형성되거나, 
    여러 군집이 하나로 합쳐질 수 있습니다.

    min_samples: 군집을 형성하기 위해 이웃 영역에 있어야 하는 최소 데이터 포인트 수를 지정합니다. 
    min_samples를 크게 설정하면, 더 밀집된 영역만 군집으로 인식되므로 
    군집의 수가 줄어들고 더 많은 데이터가 노이즈로 분류될 수 있습니다.

Q3. DBSCAN 결과에서 labels_ 속성의 값이 -1인 데이터 포인트는 무엇을 의미하나요?
    
    DBSCAN에서 labels_가 -1인 데이터 포인트는 노이즈(Noise) 또는 **이상치(Outlier)**를 의미합니다. 
    이는 어떤 군집에도 속하지 않는, 주변에 충분한 밀집도를 가진 이웃이 없는 데이터 포인트를 나타냅니다.

Q4. K-Means와 달리 DBSCAN이 '초승달' 모양의 데이터셋을 더 효과적으로 군집화할 수 있는 이유는 무엇인가요?
    
    K-Means는 구형(spherical) 군집을 가정하고, 각 군집의 중심점과 데이터 포인트 간의 거리를 기준으로 군집을 나눕니다. 
    따라서 초승달처럼 복잡하고 비선형적인 형태의 군집은 잘 분리하지 못하고, 
    오히려 하나의 큰 구형 군집으로 묶어버리는 경향이 있습니다.
    
    **DBSCAN**은 데이터의 밀집도를 기준으로 군집을 연결하므로, 군집의 형태에 대한 가정이 없습니다. 
    초승달과 같이 복잡한 형태를 가진 군집도 그 밀집된 영역을 따라 자연스럽게 하나의 군집으로 인식할 수 있어 더 효과적입니다.
"""
