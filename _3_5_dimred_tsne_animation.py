# 1. 필요한 라이브러리 임포트
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# 2. 예제 데이터 로드 및 전처리
digits = load_digits()
X = digits.data
y = digits.target

# 데이터 표준화: t-SNE는 스케일에 민감하므로 필수적입니다.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. t-SNE 초기화
# max_iter를 최소값인 250으로 설정합니다.
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    max_iter=250,  # 최소 허용값인 250으로 변경
    init="pca",
    random_state=42,
)

# 4. 애니메이션 생성
# t-SNE는 한 번의 fit_transform으로 최종 결과에 도달하므로,
# 여러 중간 결과를 시뮬레이션하여 애니메이션을 만듭니다.
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title("t-SNE Learning Process (Simulated)")
ax.set_xlabel("t-SNE Component 1")
ax.set_ylabel("t-SNE Component 2")

# 초기 위치 설정 (PCA를 사용하여 첫 프레임의 위치를 결정)
X_tsne_init = PCA(n_components=2).fit_transform(X_scaled)
scatter = ax.scatter(X_tsne_init[:, 0], X_tsne_init[:, 1], c=y, cmap="Paired", s=30)

# t-SNE 전체 학습 결과를 미리 계산합니다.
full_tsne_result = tsne.fit_transform(X_scaled)

# 프레임 수를 설정합니다 (총 1500 프레임).
# 1500프레임 * 200ms = 300초 (5분)로 너무 길어집니다.
# 150프레임 * 200ms = 30초로 동일하게 맞춥니다.
num_frames = 150

# 애니메이션의 각 프레임에 해당하는 점의 위치를 계산합니다.
# 초기 PCA 결과에서 최종 t-SNE 결과까지 선형 보간하여 중간 단계를 시뮬레이션합니다.
# 이 방법은 실제 t-SNE의 움직임과 100% 동일하지는 않지만,
# 클러스터가 형성되고 분리되는 과정을 시각적으로 명확하게 보여줍니다.
frames = []
for i in range(num_frames):
    alpha = i / (num_frames - 1)
    # 현재 프레임의 위치 = (1-알파) * PCA위치 + 알파 * t-SNE 최종 위치
    current_frame_pos = (1 - alpha) * X_tsne_init + alpha * full_tsne_result
    frames.append(current_frame_pos)


def update(frame_data):
    # 각 프레임의 데이터를 업데이트합니다.
    scatter.set_offsets(frame_data)

    return (scatter,)


# 5. 애니메이션 실행 및 저장
# 150번의 반복(프레임)으로 30초 길이의 애니메이션을 만듭니다 (150 * 0.2초 = 30초).
ani = animation.FuncAnimation(fig, update, frames=frames, interval=200)

# 애니메이션을 'tsne_animation.gif' 파일로 저장합니다. 초당 15프레임, 총 10초 분량으로 재생.
print("애니메이션을 생성하고 있습니다. 시간이 다소 소요될 수 있습니다...")
ani.save("tsne_animation.gif", writer="pillow", fps=15)
print("애니메이션 저장이 완료되었습니다. 'tsne_animation.gif' 파일을 확인해주세요.")
