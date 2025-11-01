# 1. Import necessary libraries
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch

# 2. Load example data and preprocess
digits: Bunch = load_digits()  # type: ignore
X = digits.data
y = digits.target

# Data Standardization: Essential because t-SNE is sensitive to scale.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. t-SNE Initialization
# Set max_iter to the minimum allowed value (250).
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    max_iter=250,  # Changed to the minimum allowed value of 250
    init="pca",
    random_state=42,
)

# 4. Animation Generation
# Since t-SNE reaches the final result in a single fit_transform call,
# we simulate multiple intermediate results to create the animation.
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title("t-SNE Learning Process (Simulated)")
ax.set_xlabel("t-SNE Component 1")
ax.set_ylabel("t-SNE Component 2")

# Set initial position (using PCA to determine the position of the first frame)
X_tsne_init = PCA(n_components=2).fit_transform(X_scaled)
scatter = ax.scatter(X_tsne_init[:, 0], X_tsne_init[:, 1], c=y, cmap="Paired", s=30)

# Pre-calculate the full t-SNE learning result.
full_tsne_result = tsne.fit_transform(X_scaled)

# Set the number of frames (simulating a total of 1500 frames).
# 1500 frames * 200ms = 300 seconds (5 minutes), which is too long.
# We will use 150 frames * 200ms = 30 seconds for a manageable duration.
num_frames = 150

# Calculate the position of the points corresponding to each frame of the animation.
# We linearly interpolate from the initial PCA result to the final t-SNE result to simulate intermediate steps.
# Although this method is not 100% identical to the actual movement of t-SNE,
# it visually and clearly demonstrates the process of clusters forming and separating.
frames = []
for i in range(num_frames):
    alpha = i / (num_frames - 1)
    # Current frame position = (1 - alpha) * PCA position + alpha * t-SNE final position
    current_frame_pos = (1 - alpha) * X_tsne_init + alpha * full_tsne_result
    frames.append(current_frame_pos)


def update(frame_data):
    # Update the data for each frame.
    scatter.set_offsets(frame_data)

    return (scatter,)


# 5. Run and Save the Animation
# Create a 30-second animation with 150 iterations (frames) (150 * 0.2 seconds = 30 seconds).
ani = animation.FuncAnimation(fig, update, frames=frames, interval=200)

# Save the animation to 'tsne_animation.gif'. Playing at 15 frames per second for a total duration of 10 seconds.
print("Generating animation. This may take some time...")
ani.save("tsne_animation.gif", writer="pillow", fps=15)
print("Animation saved successfully. Please check the 'tsne_animation.gif' file.")
