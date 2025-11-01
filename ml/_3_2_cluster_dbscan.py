"""
Unsupervised Learning:
    Clustering: Used to find the structure or pattern in data and group similar data points together.
        Examples: sklearn.cluster.KMeans, sklearn.cluster.DBSCAN
    Dimensionality Reduction: Reduces the number of features in data to facilitate analysis and visualization.
        Examples: sklearn.decomposition.PCA, sklearn.manifold.TSNE

DBSCAN:
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm.
    It finds clusters based on the **density** of the data points, without the need to specify the number of clusters beforehand.
    It also has the distinct feature of being able to automatically detect **Outliers** (Noise) that do not belong to any cluster.
"""

# 1. Import necessary libraries
import matplotlib.pyplot as plt  # Visualization library
import numpy as np
from sklearn.cluster import DBSCAN  # DBSCAN clustering model
from sklearn.datasets import make_moons  # Generate virtual data for clustering

# 2. Generate example data
# The make_moons function generates crescent-shaped (moon-shaped) data.
# This type of data is typically not well-separated by distance-based algorithms like KMeans.
# n_samples: The number of data samples to generate
# noise: Noise (random fluctuation) to be added to the data
# random_state: Seed value for reproducible results
X, y = make_moons(n_samples=200, noise=0.05, random_state=42)


# 3. Create Estimator Instance
# Create the DBSCAN model.
# eps (epsilon): The maximum distance to be considered a neighbor. Data points closer than this value are likely to be considered part of the same cluster.
# min_samples: The minimum number of data points required to form a cluster.
dbscan = DBSCAN(eps=0.3, min_samples=5)

# 4. Model Training (fit)
# Use the model.fit(X) method to train the model.
# Since it's unsupervised learning, only the feature data (X) is used, without the correct labels (y).
# Through this process, DBSCAN finds the densely packed areas of the data to form clusters.
print("Starting DBSCAN clustering...")
dbscan.fit(X)
print("DBSCAN clustering complete!")

# 5. Cluster Label Assignment
# After the fit() method runs, the cluster label assigned to each data point is stored in the labels_ attribute.
# -1 signifies noise (outliers), and non-negative integers starting from 0 represent each cluster.
labels = dbscan.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

# 6. Visualize Results
plt.figure(figsize=(8, 6))
# Visualize each cluster using a different color.
# Outliers (-1) are colored black.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Color outliers black
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

# 7. Output Cluster Information
print(f"\nEstimated number of clusters (excluding noise): {n_clusters_}")
print(f"Number of data points classified as noise: {n_noise_}")

"""
Execution Example

Starting DBSCAN clustering...
DBSCAN clustering complete!

Estimated number of clusters (excluding noise): 2
Number of data points classified as noise: 0
"""

"""
Review

Q1. Unlike KMeans, DBSCAN does not require the number of clusters to be specified beforehand. What characteristic of DBSCAN makes this possible?

    DBSCAN forms clusters based on **density**.
    If data points are sufficiently dense in a particular region, it is considered a single cluster.
    Therefore, the user does not need to pre-determine the number of clusters (K);
    the algorithm analyzes the data's density itself to determine the number of clusters.

Q2. What are the roles of the main parameters of DBSCAN, eps and min_samples, and how does adjusting them affect the result?

    **eps (epsilon)**: Specifies the maximum distance to be considered a neighbor from a data point.
    Setting a larger eps includes more data points as neighbors, which can lead to the formation of larger clusters or merge multiple clusters into one.

    **min_samples**: Specifies the minimum number of data points required in the neighborhood to form a cluster.
    Setting a larger min_samples recognizes only denser areas as clusters, which can decrease the number of clusters and classify more data points as noise.

Q3. What does it mean when a data point in the DBSCAN result has a value of -1 in the labels_ attribute?

    In DBSCAN, a data point with a `labels_` value of -1 signifies **Noise** or an **Outlier**.
    This indicates a data point that does not belong to any cluster, as it does not have enough densely packed neighbors in its vicinity.

Q4. Why can DBSCAN cluster a 'moon-shaped' dataset more effectively than K-Means?

    K-Means assumes **spherical** clusters and divides them based on the distance between the cluster center and the data points.
    Therefore, it struggles to separate complex, non-linear cluster shapes like the moon shape,
    often tending to group them into one large, spherical cluster.

    **DBSCAN** connects clusters based on data **density** and has no assumption about the shape of the clusters.
    It can follow the densely packed area, naturally recognizing complex shapes like the moon shape as a single cluster, making it more effective.
"""
