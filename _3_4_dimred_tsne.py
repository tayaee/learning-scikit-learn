"""
비지도 학습 (Unsupervised Learning):
    군집 (Clustering): 데이터의 구조나 패턴을 찾아 유사한 데이터끼리 그룹화하는 데 사용됩니다.
        예: sklearn.cluster.KMeans, sklearn.cluster.DBSCAN
    차원 축소 (Dimensionality Reduction): 데이터의 차원을 줄여 분석 및 시각화를 용이하게 합니다.
        예: sklearn.decomposition.PCA, sklearn.manifold.TSNE

t-SNE:
    t-SNE(t-Distributed Stochastic Neighbor Embedding)는 고차원 데이터를
    2D 또는 3D와 같은 저차원으로 시각화하는 데 매우 효과적인 비지도 학습 알고리즘입니다.
    데이터 포인트 간의 거리와 유사성을 보존하면서 시각적으로 군집을 쉽게 파악할 수 있도록
    데이터를 재배치하는 특징이 있습니다.
"""

# 1. 필요한 라이브러리 임포트
import matplotlib.pyplot as plt  # 시각화 라이브러리
from sklearn.datasets import load_digits  # 예제 데이터셋 (손글씨 숫자)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE  # t-SNE 차원 축소 모델
from sklearn.preprocessing import StandardScaler  # 데이터 스케일링

# 2. 예제 데이터 로드 및 전처리
# t-SNE는 PCA와 마찬가지로 스케일에 민감하므로 데이터를 표준화하는 것이 좋습니다.
digits = load_digits()
X = digits.data  # 8x8 크기의 이미지 데이터를 64차원 벡터로 변환한 데이터
y = digits.target  # PCA와 마찬가지로 시각화를 위해 타겟(y)을 사용합니다.

# ==============================================================================
# 섹션 2-1: 원본 데이터 시각화 (참고용)
# t-SNE로 차원 축소하기 전의 원본 데이터가 어떤 모습인지 확인합니다.
# 여기서는 손글씨 숫자 이미지 몇 개를 실제 이미지 형태로 시각화합니다.
# ==============================================================================
fig, axes = plt.subplots(2, 5, figsize=(10, 5), subplot_kw={"xticks": [], "yticks": []})
for i, ax in enumerate(axes.flat):
    # digits.images는 8x8 이미지 형태의 데이터를 담고 있습니다.
    ax.imshow(digits.images[i], cmap="binary", interpolation="nearest")
    ax.set_title(f"Label: {digits.target[i]}")
plt.suptitle("Visualization of Original Digits Data", fontsize=16)
plt.show()

# 데이터 표준화: 평균을 0, 표준편차를 1로 만듭니다.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 추정기(Estimator) 인스턴스 생성
# t-SNE 모델을 생성합니다.
# n_components: 축소할 차원의 수. 일반적으로 2 또는 3으로 설정하여 시각화에 사용합니다.
# perplexity: 각 데이터 포인트의 이웃을 몇 개까지 고려할지를 나타내는 값.
# 데이터의 구조에 따라 최적의 값이 달라질 수 있습니다.
# n_iter: 최적화를 위한 반복 횟수. 충분히 큰 값으로 설정해야 안정적인 결과를 얻을 수 있습니다.
# learning_rate: 학습률. 너무 작으면 수렴이 느리고, 너무 크면 불안정할 수 있습니다.
tsne = TSNE(n_components=2, perplexity=30, max_iter=300, random_state=42)

# 4. 모델 학습 및 변환 (fit_transform)
# model.fit_transform(X) 메서드를 사용하여 모델을 학습시키고 데이터를 변환합니다.
# t-SNE는 비지도 학습이므로, 정답 레이블(y) 없이 특징 데이터(X)만 사용합니다.
print("t-SNE 차원 축소 시작...")
X_tsne = tsne.fit_transform(X_scaled)
print("t-SNE 차원 축소 완료!")
print("\n원래 데이터 차원:", X_scaled.shape)
print("t-SNE 적용 후 데이터 차원:", X_tsne.shape)

# 5. 결과 시각화
# 2차원으로 축소된 데이터를 시각화하여 군집 구조를 파악합니다.
# 각 점의 색상은 원래의 숫자 레이블을 나타냅니다.
# 그림 상의 점 한 개는 64차원 공간에 있던 손글씨 숫자 이미지 한 개에 해당합니다.
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="Paired", s=30)
plt.title("t-SNE Visualization of Digits Dataset")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.colorbar(label="Digit Class")

# ==============================================================================
# 섹션 6: PCA와 t-SNE 결과 비교
# 동일한 데이터에 PCA를 적용하여 t-SNE와 어떻게 다른지 비교합니다.
# ==============================================================================
print("\nPCA 차원 축소 시작...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print("PCA 차원 축소 완료!")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="Paired", s=30)
axes[0].set_title("t-SNE Visualization")
axes[0].set_xlabel("t-SNE Component 1")
axes[0].set_ylabel("t-SNE Component 2")

axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="Paired", s=30)
axes[1].set_title("PCA Visualization")
axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")

plt.suptitle("t-SNE vs PCA for Digits Dataset Visualization", fontsize=16)
plt.show()

"""
실행 예:

t-SNE 차원 축소 시작...
t-SNE 차원 축소 완료!

원래 데이터 차원: (1797, 64)
t-SNE 적용 후 데이터 차원: (1797, 2)
"""

"""
복습

Q1. t-SNE는 PCA와 같은 차원 축소 알고리즘이지만, 주로 어떤 목적으로 사용되나요? 두 알고리즘의 가장 큰 차이점은 무엇인가요?

    t-SNE는 고차원 데이터의 시각화를 목적으로 주로 사용됩니다. 
    PCA는 특히, 데이터 포인트 간의 지역적(local) 구조를 보존하면서 
    저차원으로 변환하는 데 뛰어나 복잡한 데이터의 군집 구조를 한눈에 파악하기 용이합니다.

    주요 차이점: PCA는 전역적(global) 구조를 보존하며 데이터의 분산을 최대화하는 방향으로 차원을 축소합니다. 
    반면, t-SNE는 지역적 구조를 보존하며 데이터 포인트 간의 유사성을 저차원에서 잘 표현하는 데 초점을 맞춥니다.

Q2. t-SNE 모델에서 perplexity 매개변수의 역할은 무엇이며, 이 값이 결과에 어떻게 영향을 미치나요?
    
    **perplexity**는 한 데이터 포인트가 **"얼마나 많은 이웃을 고려해야 하는가"**를 정의하는 값입니다. 
    이 값은 데이터의 지역적, 전역적 구조를 모두 고려하는 균형점을 찾는 데 중요합니다.
    
    perplexity 값이 작으면: 지역적 구조에 더 집중하게 되어 작은 군집들이 여러 개로 나뉠 수 있습니다.
    
    perplexity 값이 크면: 전역적 구조를 더 많이 고려하게 되어 큰 군집들이 형성되고, 
    복잡한 데이터의 경우 과도하게 뭉쳐 보일 수 있습니다.

Q3. t-SNE를 적용하기 전에 데이터를 StandardScaler로 표준화하는 이유는 무엇인가요?
    
    t-SNE는 데이터 포인트 간의 거리를 기반으로 유사성을 계산합니다. 
    따라서 각 특징(feature)의 스케일이 크게 다르면, 
    스케일이 큰 특징이 거리 계산에 더 큰 영향을 미쳐서 결과가 왜곡될 수 있습니다. 
    StandardScaler를 사용해 모든 특징의 스케일을 동일하게 맞춰주면, 
    모든 특징이 동등한 중요도를 가지게 되어 더 정확한 시각화 결과를 얻을 수 있습니다.

Q4. t-SNE 시각화 결과에서 군집들이 명확하게 구분되는 것을 확인할 수 있습니다. 이것이 의미하는 바는 무엇인가요?
    
    이는 고차원 데이터 공간에서 동일한 레이블을 가진 데이터 포인트들
    (예: 같은 숫자 이미지들)이 서로 가깝게 위치하고, 
    다른 레이블을 가진 데이터 포인트들과는 멀리 떨어져 있다는 것을 의미합니다. 
    다시 말해, t-SNE가 고차원 공간에서의 복잡한 관계를 저차원에서도 성공적으로 보존했음을 보여주는 결과입니다.

Q5. 차원 축소는 그러면 시각화가 주 목적의 하나인가?

    네, 차원 축소의 주요 목적 중 하나는 시각화입니다.

Q6. PCA대비 t-SNE가 더 좋다는 것인가? 항상 더 좋은가? 더 좋을 때도 있는 것인가. 선택 기준은?

    t-SNE가 항상 PCA보다 더 좋은 것은 아닙니다.
    t-SNE는 특히 데이터의 군집 구조를 시각화하는 데 뛰어나지만,
    계산 비용이 높고, 대규모 데이터셋에는 적합하지 않을 수 있습니다.
    PCA는 계산이 빠르고, 데이터의 분산을 최대화하는 방향으로 차원을 축소하기 때문에,
    데이터의 전반적인 구조를 이해하는 데 유용합니다.
    선택 기준은 분석 목적, 데이터 크기, 계산 자원 등에 따라 달라집니다.
    이 숫자 인식 예에서는 t-SNE가 더 좋은 결과를 보여주네요.

Q7. PCA, t-SNE 둘 다 해봐야 하나?

    네, PCA와 t-SNE는 서로 보완적인 차원 축소 기법이므로, 둘 다 시도해보는 것이 좋습니다.

Q8. 왜 t 분포인가?

    SNE는 정규분포를 사용했는데 그것의 Crowding Problem을 해결하기 위해 t 분포를 사용합니다.
    t 분포는 정규분포보다 꼬리가 두꺼워, 고차원 데이터의 지역적 구조를 더 잘 보존할 수 있습니다.

Q9. SNE의 목적 함수를 먼저 이해해보자.

    SNE의 목적 함수는 데이터 포인트 간의 유사성을 보존하는 것입니다.
    SNE는 고차원 공간(이 예저의 경우 64차원)에서의 데이터 포인트 간의 거리를 
    저차원 (이 예제의 경우 2차원) 공간에서도 최대한 보존하려고 합니다.
    이를 위해, SNE는 각 데이터 포인트의 이웃과의 거리를 확률적으로 모델링하고, 
    이 확률 분포를 최소화하는 방향으로 최적화합니다.
    t-SNE는 SNE의 목적 함수를 개선하여, 지역적 구조를 더 잘 보존하도록 설계되었습니다.
   
    SNE의 핵심은 고차원 공간의 확률 분포와 저차원 공간의 확률 분포를 최대한 비슷하게 만드는 것입니다. 
    이 "유사성"을 측정하는 지표로 
    **쿨백-라이블러 발산(Kullback-Leibler Divergence, KL Divergence)**이라는 것을 사용합니다.
    SNE의 목적 함수는 다음과 같고 SNE는 C값을 최소화 하도록 점들의 위치를 조정합니다.

    \[
    C = \sum_{i} D_{KL}(P_i || Q_i) = \sum_{i} \sum_{j} P_{ij} \log \frac{P_{ij}}{Q_{ij}}
    \]

    여기서 \( P_{ij} \)는 고차원 공간에서의 데이터 포인트 \( i \)와 \( j \) 
    사이의 유사성 확률(고차원 공간에서 xi가 xj를 이웃으로 선택할 확률)이고,
    \( Q_{ij} \)는 저차원 공간에서의 유사성 확률입니다.
    SNE는 이 KL 발산을 최소화하여, 고차원 공간의 구조를 저차원 공간에서도 잘 보존하려고 합니다.
    KL발산이 0에 가까울수록 두 확률 분포는 유사하다는 것을 의미합니다.

Q10. 그래서 SNE에서 Crowding Problem 이란 게 뭐야.

    목적 함수 내 항들인 P_ij와 Q_ij를 계산할 때 가우스 분포를 사용합니다.

    고차원에서 가까운 점들이 저차원에서 너무 가깝게 위치하게 되는 건 문제가 덜 되지만
    고차원에서 멀리 떨어진 점들이 저차원에서 너무 가까워지는 문제(즉 오류)가 발생합니다.
    예를 들면 1과 7의 두 개 클러스터들은 고차원에서 멀리 떨어져 있지만
    SNE를 사용할 경우 저차원에서는 이 둘이 합쳐지는 문제가 발생합니다.
    이 문제를 Crowding Problem이라고 합니다.

    t-SNE에서는 이 문제를 t분포를 사용하여
    고차원에서 멀리 떨어져 있던 점들의 거리를
    저차원에서도 적절히 유지하도록 개선했습니다.

    SNE는 2002년 발표.
    t-SNE는 2008년 발표.

Q11. SNE가 정규 분포를 사용한다는 의미는 뭐야.

    SNE의 공식은 1차원 정규분포의 PDF(누적 분포 함수)를 사용합니다.

    1차원 정규분포의 PDF는 다음과 같습니다:
    \[
    P(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}
    \]

    이중 exp 부분의 모양이 정확히 SNE의 목적 함수에서 P_ij를 계산하는 방식과 유사합니다.
    그 부분이 정규분포의 종모양 (벨 커브)를 형성합니다.
    SNE 공식의 분자는 정규분포 PDF의 핵심적인 종 모양 곡선을 그대로 따르고 있습니다.

    따라서 SNE는 두 점 사이의 유사성을 정규분포의 형태를 빌려 확률로 표현했다고 할 수 있습니다.

Q12. 목적 함수에 sigma^2 즉 정규분포의 분산이 들어가는데, 이것은 입력 데이터가 
    정규분포이기 때문에 이 값을 계산하는 것인가 아니면 거리값들이 정규분포임을 가정하고 
    이 값을 계산하는 것인가?

    '정규 분포를 가정하고 분산을 계산했다'는 표현이 정확합니다.
    증명된 것이 아닙니다.

Q13. n_components=2, perplexity=30 파라미터 선정 best practice는?

    n_components=2는 일반적으로 시각화를 위해 많이 사용되는 값입니다.
    2차원으로 축소하면 2D 플롯에서 데이터를 시각화할 수 있어 직관적입니다.
    3차원으로 축소할 수도 있지만, 2D가 더 일반적입니다.

    perplexity는 데이터의 밀도와 구조에 따라 달라집니다.
    일반적으로 5에서 50 사이의 값을 사용합니다.
    perplexity가 작으면 지역적 구조에 더 집중하고,
    perplexity가 크면 전역적 구조에 더 집중합니다.
    따라서 데이터의 특성에 따라 최적의 perplexity 값을 찾는 것이 중요합니다.

    일반적으로 perplexity는 데이터의 크기와 밀도에 따라 조정됩니다.
    예를 들어, 데이터가 밀집되어 있다면 작은 perplexity 값을 사용하고,
    데이터가 희박하다면 큰 perplexity 값을 사용하는 것이 좋습니다.
    일반적으로 5에서 50 사이의 값을 사용하며, 데이터에 따라 최적의 값을 찾는 것이 중요합니다.

Q14. SNE와 t-SNE의 목적함수 차이를 비교해줘.

    SNE에서는 두 점이 이웃일 확률을 계산하기 위해
        고차원에서 P_ij
        저차원에서 Q_ij
        둘 다 가우스분포 공식을 사용.

    t-SNE에서는 두 점이 이웃일 확률을 계산하기 위해
        고차원에서 P_ij 를 계산하기 위해서는 SNE와 동일하게 가우스 분포 사용
        저차원에서 Q_ij 를 계산하기 위해 꼬리가 두터운 t 분포 사용

    목적 함수 C의 모양은 둘이 똑같지만, 함수 내 분모에 들어가는 저차원 공간에서의 유사성 확률 계산 공식이 t 분포 모양으로 대체됨.
"""
