"""
비지도 학습 (Unsupervised Learning):
    군집 (Clustering): 데이터의 구조나 패턴을 찾아 유사한 데이터끼리 그룹화하는 데 사용됩니다.
        예: sklearn.cluster.KMeans, sklearn.cluster.DBSCAN
    차원 축소 (Dimensionality Reduction): 데이터의 차원을 줄여 분석 및 시각화를 용이하게 합니다.
        예: sklearn.decomposition.PCA, sklearn.manifold.TSNE

PCA:
    PCA (Principal Component Analysis)는 데이터의 차원을 축소하는 가장 대표적인 비지도 학습 알고리즘입니다.
    데이터가 가진 원래의 분산(Variance)을 최대한 보존하면서,
    상관관계가 없는 새로운 축(주성분)을 찾아 데이터를 투영하여 차원을 줄이는 데 사용됩니다.
"""

# 1. 필요한 라이브러리 임포트
import time

import matplotlib.pyplot as plt  # 시각화 라이브러리
import pandas as pd
from sklearn.datasets import load_iris  # 예제 데이터셋 (붓꽃)
from sklearn.decomposition import PCA  # PCA 차원 축소 모델
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 데이터 스케일링

# 2. 예제 데이터 로드 및 전처리
# PCA는 특징(feature)의 스케일에 민감하므로, 데이터를 표준화하는 것이 중요합니다.
iris = load_iris()
X = iris.data
y = iris.target  # PCA는 비지도 학습이지만, 결과를 시각화하기 위해 y(타겟)를 사용합니다.

# 데이터 분할: 모델의 일반화 성능을 평가하기 위해 먼저 데이터를 분할합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ==============================================================================
# 섹션 2-1: 원본 데이터 시각화 (참고용)
# 4차원 데이터를 2차원 평면에 직접 시각화할 수 없으므로,
# 대표적인 두 특징 쌍(꽃받침/꽃잎의 길이/너비)을 선택하여 시각화합니다.
# 이를 통해 PCA가 왜 필요한지 직관적으로 이해할 수 있습니다.
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 꽃받침(Sepal) 길이 vs 너비
axes[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="viridis", edgecolor="k", s=50)
axes[0].set_title("Original Data (Sepal Features)")
axes[0].set_xlabel(iris.feature_names[0])
axes[0].set_ylabel(iris.feature_names[1])
axes[0].grid(True)

# 꽃잎(Petal) 길이 vs 너비
axes[1].scatter(X_train[:, 2], X_train[:, 3], c=y_train, cmap="viridis", edgecolor="k", s=50)
axes[1].set_title("Original Data (Petal Features)")
axes[1].set_xlabel(iris.feature_names[2])
axes[1].set_ylabel(iris.feature_names[3])
axes[1].grid(True)

plt.suptitle("Visualization of Original 4D Iris Data using Feature Pairs", fontsize=16)
plt.show()

# PCA가 필요한 이유
# 꽃잎(길이, 너비)와 꽃받침(길이, 너비)의 특징 쌍을 즉 4차원 데이터를 시각화하면
# 데이터가 서로 겹치거나 분포가 복잡하여, 군집이나 패턴을 파악하기 어렵습니다.
# setosa (보라색)은 잘 분리가 되지만 versicolor와 virginica는 겹칩니다.
# PCA를 통해 이 데이터를 2차원으로 축소하면, 서로 다른 클래스(붓꽃 종류)를 더 명확하게 구분할 수 있습니다.


# 데이터 표준화: 평균을 0, 표준편차를 1로 만듭니다.
# 중요: Scaler는 학습 데이터(X_train)에만 fit 해야 합니다.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 테스트 데이터는 transform만 적용

# 3. 추정기(Estimator) 인스턴스 생성
# PCA 모델을 생성합니다.
# n_components: 축소할 차원의 수 (주성분 개수)를 지정합니다.
# 붓꽃 데이터는 4차원이지만, 2차원으로 축소하여 시각화하기 쉽게 만듭니다.
pca = PCA(n_components=2)

# 4. 모델 학습 및 변환 (fit_transform)
# model.fit_transform(X) 메서드를 사용하여 모델을 학습시키고 데이터를 변환합니다.
# 중요: PCA도 학습 데이터(X_train_scaled)에만 fit 해야 합니다.
print("PCA 차원 축소 시작...")
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)  # 테스트 데이터는 transform만 적용
print("PCA 차원 축소 완료!")
print("\n학습 데이터 차원 (원본):", X_train_scaled.shape)
print("학습 데이터 차원 (PCA):", X_train_pca.shape)
print("테스트 데이터 차원 (PCA):", X_test_pca.shape)

# 5. 결과 시각화
# 2차원으로 축소된 데이터를 시각화합니다.
# 원래의 타겟(y) 레이블을 사용하여 시각화하면, PCA가 얼마나 잘 분산 정보를 보존했는지 알 수 있습니다.
plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap="viridis", edgecolor="k", s=50)
plt.title("PCA Result of Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Class")
plt.show()

# 6. 주성분 설명력 출력
# 각 주성분이 설명하는 분산의 비율을 확인합니다.
explained_variance_ratio = pca.explained_variance_ratio_
print(f"각 주성분의 설명된 분산 비율: {explained_variance_ratio}")
print(f"총 설명된 분산: {sum(explained_variance_ratio):.2f}")

# 7. 주성분과 원본 특성 간의 관계 확인
# pca.components_ 속성은 각 주성분이 어떤 원본 특성들의 조합으로 이루어졌는지 보여줍니다.
# 행: 주성분 (PC1, PC2)
# 열: 원본 특성 (sepal length, sepal width, petal length, petal width)
print("\n--- 주성분과 원본 특성 간의 관계 ---")
components_df = pd.DataFrame(
    pca.components_,
    columns=iris.feature_names,
    index=["Principal Component 1", "Principal Component 2"],
)
print("주성분별 원본 특성의 기여도(가중치):")
print(components_df)
(
    "해석: 예를 들어, Principal Component 1은 모든 특성이 양의 방향으로 기여하며, "
    "특히 'petal length'와 'petal width'의 영향이 큽니다."
    "Principal Component 2는 'sepal length'와 'sepal width'의 영향이 상대적으로 큽니다."
)

# ==============================================================================
# 섹션 8: PCA의 실용적 활용 예시 (모델 성능 및 속도 비교)
# PCA로 차원 축소된 데이터가 실제 모델 학습에 어떻게 사용되는지 보여줍니다.
# 원본 데이터와 PCA 데이터를 각각 사용하여 로지스틱 회귀 모델을 학습시키고,
# 정확도와 학습 시간을 비교합니다.
# ==============================================================================

print("\n--- [활용 예시] 모델 성능 및 속도 비교 ---")

# 1. 원본 데이터(4차원)로 모델 학습 및 예측
start_time = time.time()
model_original = LogisticRegression(random_state=42)
model_original.fit(X_train_scaled, y_train)
duration_original = time.time() - start_time

# 테스트 데이터로 예측 및 평가
y_pred_original = model_original.predict(X_test_scaled)
accuracy_original = model_original.score(X_test_scaled, y_test)
print(f"원본 데이터(4D) - 학습 시간: {duration_original:.6f} 초")
print(f"원본 데이터(4D) - 테스트 정확도: {accuracy_original:.4f}")

# 2. PCA 데이터(2차원)로 모델 학습 및 예측
start_time = time.time()
model_pca = LogisticRegression(random_state=42)
model_pca.fit(X_train_pca, y_train)
duration_pca = time.time() - start_time

# 테스트 데이터로 예측 및 평가
y_pred_pca = model_pca.predict(X_test_pca)
accuracy_pca = model_pca.score(X_test_pca, y_test)
print(f"\nPCA 데이터(2D) - 학습 시간: {duration_pca:.6f} 초")
print(f"PCA 데이터(2D) - 테스트 정확도: {accuracy_pca:.4f}")

print(
    "\n결론: 붓꽃 데이터는 작아서 차이가 미미하지만, PCA를 통해 특징의 수를 절반으로 줄여도\n      정확도는 거의 유지하면서 학습 속도를 개선할 수 있습니다. (대규모 데이터에서 효과 극대화)"
)
print("-" * 50)

# ==============================================================================
# 섹션 9: 새로운 데이터에 대한 예측 과정
# 학습된 Scaler, PCA, 모델을 사용하여 완전히 새로운 데이터 1개를 예측하는 과정입니다.
# ==============================================================================
print("\n--- [활용 예시] 새로운 데이터 1개 예측 ---")

# 새로운 데이터 샘플 (4차원)
new_sample = [[5.9, 3.0, 5.1, 1.8]]  # Virginica 품종에 가까운 데이터
print(f"1. 새로운 원본 데이터: {new_sample} (Virginica 품종에 가까운 데이터)")

# 2. 학습에 사용된 Scaler로 변환
new_sample_scaled = scaler.transform(new_sample)
print(f"2. 스케일링된 데이터: {new_sample_scaled}")

# 3. 학습에 사용된 PCA로 변환
new_sample_pca = pca.transform(new_sample_scaled)
print(f"3. PCA 변환된 데이터: {new_sample_pca}")

# 4. 최종 모델로 예측
prediction_original = model_original.predict(new_sample_scaled)
prediction_pca = model_pca.predict(new_sample_pca)

print(f"\n5. 원본 데이터 모델 예측 결과: {iris.target_names[prediction_original[0]]}")
print(f"   PCA 데이터 모델 예측 결과: {iris.target_names[prediction_pca[0]]}")

"""
PCA 차원 축소 시작...
PCA 차원 축소 완료!

학습 데이터 차원 (원본): (105, 4)
학습 데이터 차원 (PCA): (105, 2)
테스트 데이터 차원 (PCA): (45, 2)
각 주성분의 설명된 분산 비율: [0.7264421  0.23378786]
총 설명된 분산: 0.96

--- 주성분과 원본 특성 간의 관계 ---
주성분별 원본 특성의 기여도(가중치):
                       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
Principal Component 1           0.530568         -0.240617           0.582340          0.566993
Principal Component 2           0.337784          0.939166           0.024723          0.057082

--- [활용 예시] 모델 성능 및 속도 비교 ---
원본 데이터(4D) - 학습 시간: 0.017752 초
원본 데이터(4D) - 테스트 정확도: 0.9111

PCA 데이터(2D) - 학습 시간: 0.011999 초
PCA 데이터(2D) - 테스트 정확도: 0.8889

결론: 붓꽃 데이터는 작아서 차이가 미미하지만, PCA를 통해 특징의 수를 절반으로 줄여도
      정확도는 거의 유지하면서 학습 속도를 개선할 수 있습니다. (대규모 데이터에서 효과 극대화)
--------------------------------------------------

--- [활용 예시] 새로운 데이터 1개 예측 ---
1. 새로운 원본 데이터: [[5.9, 3.0, 5.1, 1.8]] (Virginica 품종에 가까운 데이터)
2. 스케일링된 데이터: [[ 0.0310503  -0.12139684  0.74075533  0.76797223]]
3. PCA 변환된 데이터: [[ 0.91249102 -0.0413724 ]]

5. 원본 데이터 모델 예측 결과: virginica
   PCA 데이터 모델 예측 결과: versicolor    <== 틀렸네
"""

"""
복습

Q1. PCA는 왜 비지도 학습에 속하나요? fit_transform() 메서드에 y를 사용하지 않는 것과 어떤 관계가 있나요?

    PCA는 데이터의 **내재된 구조(분산)**를 분석하여 차원을 축소하는 알고리즘입니다. 
    이 과정에서 정답 레이블(y)의 정보는 전혀 필요하지 않습니다. 
    따라서 fit_transform() 메서드에 y를 사용하지 않으며, 비지도 학습으로 분류됩니다.

Q2. PCA를 적용하기 전에 StandardScaler를 사용하는 이유는 무엇인가요? 이 과정을 생략하면 어떤 문제가 발생할 수 있나요?

    PCA는 데이터의 분산을 기준으로 주성분을 찾습니다. 
    만약 데이터의 특징(feature)별로 스케일이 크게 다르다면, 
    분산이 큰 특징이 분산이 작은 특징보다 더 중요하게 간주될 수 있습니다.

    StandardScaler를 사용해 모든 특징의 스케일을 통일하면, 
    PCA가 특정 특징에 편향되지 않고 데이터의 실제 분산 구조를 공정하게 파악할 수 있게 됩니다. 
    이를 생략하면 결과가 왜곡될 수 있습니다.

Q3. PCA의 주요 매개변수인 n_components의 역할은 무엇인가요?
    
    n_components는 PCA를 통해 축소할 차원의 수 또는 새롭게 생성할 주성분의 개수를 지정하는 매개변수입니다. 
    예를 들어 n_components=2로 설정하면, 원래의 N차원 데이터를 
    가장 분산이 큰 2개의 주성분을 가진 2차원 데이터로 변환합니다.

Q4. pca.explained_variance_ratio_ 속성이 의미하는 바는 무엇이며, 이 값이 왜 중요한가요?
    
    explained_variance_ratio_는 각 주성분이 전체 데이터 분산에서 차지하는 비율을 나타냅니다. 
    예를 들어, 첫 번째 주성분(Principal Component 1)이 0.95라는 값을 가지면, 
    전체 분산의 95%를 이 주성분이 설명한다는 의미입니다.
    
    이 값은 축소된 차원이 원래 데이터의 정보를 얼마나 잘 보존하고 있는지를 평가하는 데 중요합니다. 
    이 비율을 합산하여 원하는 정보 손실 수준을 유지하면서 차원 축소를 수행할 수 있습니다.

Q5. PCA를 사용하여 차원을 축소한 후, 원본 데이터와 PCA 데이터를 각각 로지스틱 회귀 모델로 학습시킨 결과는 어땠나요?

    PCA를 사용하여 차원을 축소한 후, 원본 데이터와 PCA 데이터를 각각 로지스틱 회귀 모델로 학습시킨 결과는 다음과 같았습니다:
    - 원본 데이터(4차원)의 테스트 정확도는 약 0.9111로 나타났습니다.
    - PCA 데이터(2차원)의 테스트 정확도는 약 0.8889로 나타났습니다.
    PCA를 통해 차원을 축소했음에도 불구하고, 정확도는 거의 유지되었습니다.

Q6. PCA를 적용한 후, 새로운 데이터에 대한 예측 과정은 어떻게 진행되나요?

    PCA를 적용한 후, 새로운 데이터에 대한 예측 과정은 다음과 같이 진행됩니다:   
    1. 새로운 데이터를 원본 스케일로 입력합니다.
    2. 학습에 사용된 StandardScaler를 사용하여 데이터를 스케일링합니다.
    3. 학습에 사용된 PCA를 적용하여 데이터를 차원 축소합니다.
    4. 최종적으로 학습된 모델을 사용하여 예측을 수행합니다.
    
    위 예의 경우 새로운 데이터 [[5.9, 3.0, 5.1, 1.8]]에 대해
    원본 데이터 모델은 'virginica'로 예측했지만, PCA 데이터 모델은 'versicolor'로 예측했습니다.
    PCA 데이터 모델이 원본 데이터 모델과 다른 예측을 한 이유는
    PCA로 인해 데이터의 구조가 약간 변경되었기 때문입니다.    

Q7. ML 연구자가 PCA를 사용해야 할 거 같다고 생각하게 만드는 요인은 어떤 것이 있습니까.

    차원의 저주(Curse of Dimensionality) 문제 해결
        데이터 크기에 비해 특징(Feature)의 수가 압도적으로 많은 경우, PCA는 불필요하거나 중복된 정보를 제거하여 모델이 학습해야 할 특징의 수를 줄여줍니다.
        모델의 학습 속도 및 메모리 효율성 개선:
            대규모 데이터셋에서 수많은 특징을 모두 사용하면 학습 시간이 매우 오래 걸리고, 막대한 메모리를 소비합니다. 
            PCA를 통해 차원을 축소하면 학습 데이터의 크기가 줄어들어 훨씬 빠른 속도로 모델을 학습시킬 수 있습니다. 
            특히 딥러닝 모델의 초기 레이어에 PCA를 적용하여 특징의 수를 줄이는 경우가 있습니다.
    다중 공선성(Multicollinearity) 문제 해결:
        예를 들어, 집값을 예측할 때 '방의 개수'와 '화장실의 개수'가 높은 상관관계를 보인다면, PCA로 이를 통합하여 새로운 주성분을 만들 수 있습니다.
    데이터 시각화 및 패턴 발견
        사람의 눈은 2차원 또는 3차원 공간만 인식할 수 있습니다. 
        유전자 데이터나 고객 행동 데이터와 같이 수백, 수천 개의 특징을 가진 데이터를 시각화하는 것은 불가능합니다. 
        PCA는 이러한 고차원 데이터를 가장 중요한 2~3개의 주성분으로 축소하여 시각화함으로써 
        데이터의 군집(Clustering) 구조나 이상치(Outlier)를 직관적으로 파악할 수 있게 해줍니다. 
        이는 데이터 분석의 초기 단계에서 매우 유용합니다.
    노이즈 제거(Noise Reduction)
        PCA는 분산이 높은 방향을 중심으로 주성분을 찾습니다. 
        분산이 매우 낮은 주성분은 데이터의 노이즈나 미세한 변동성을 나타내는 경우가 많습니다. 
        이러한 분산이 낮은 주성분을 제거하고 분산이 높은 주성분만 사용하면 
        데이터의 중요한 패턴을 유지하면서 노이즈를 효과적으로 제거할 수 있습니다.
    데이터의 압축 및 저장 효율성
        대규모 고차원 데이터를 저장하거나 전송할 때, 
        PCA를 사용해 차원을 축소하면 데이터 크기를 크게 줄일 수 있습니다. 
        이는 실시간 시스템이나 클라우드 기반 서비스에서 저장 비용을 절감하는 데 도움이 됩니다.

Q8. PCA의 단점은

    정보 손실: Iris 예시처럼, 차원 축소 과정에서 분류나 회귀에 중요한 미세한 정보가 손실될 수 있습니다. 
    연구자는 explained_variance_ratio_를 확인하며 어느 정도의 정보 손실이 허용 가능한지 판단합니다.

    해석의 어려움: 새로운 주성분은 여러 원본 특징의 선형 조합이므로, 
    "주성분 1이 어떤 의미인가?"를 직관적으로 해석하기 어렵습니다. 
    모델의 결과를 비즈니스 측면에 설명해야 할 때는 PCA가 오히려 단점이 될 수 있습니다.

"""
