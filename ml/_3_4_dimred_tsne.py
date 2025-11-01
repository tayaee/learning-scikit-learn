"""
Unsupervised Learning:
    Clustering: Used to find the structure or pattern in data and group similar data points together.
        Examples: sklearn.cluster.KMeans, sklearn.cluster.DBSCAN
    Dimensionality Reduction: Reduces the dimension of data to facilitate analysis and visualization.
        Examples: sklearn.decomposition.PCA, sklearn.manifold.TSNE

t-SNE:
    t-SNE (t-Distributed Stochastic Neighbor Embedding) is an unsupervised learning algorithm highly effective for visualizing high-dimensional data
    in low dimensions, such as 2D or 3D.
    It has the characteristic of rearranging data to allow for easy visual identification of clusters while preserving the distance and similarity between data points.
"""

# 1. Import necessary libraries
import matplotlib.pyplot as plt  # Visualization library
from sklearn.datasets import load_digits  # Example dataset (handwritten digits)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE  # t-SNE dimensionality reduction model
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch  # Data utility class

# 2. Load example data and preprocess
# Like PCA, t-SNE is sensitive to scale, so it is recommended to standardize the data.
digits: Bunch = load_digits()  # type: ignore
X = digits.data  # 8x8 image data transformed into a 64-dimensional vector
y = digits.target  # Similar to PCA, the target (y) is used for visualization.

# ==============================================================================
# Section 2-1: Original Data Visualization (For Reference)
# Check what the original data looks like before dimensionality reduction with t-SNE.
# Here, a few handwritten digit images are visualized in their actual image form.
# ==============================================================================
fig, axes = plt.subplots(2, 5, figsize=(10, 5), subplot_kw={"xticks": [], "yticks": []})
for i, ax in enumerate(axes.flat):
    # digits.images contains the data in 8x8 image format.
    ax.imshow(digits.images[i], cmap="binary", interpolation="nearest")
    ax.set_title(f"Label: {digits.target[i]}")
plt.suptitle("Visualization of Original Digits Data", fontsize=16)
plt.show()

# Data Standardization: Scale the data to have a mean of 0 and a standard deviation of 1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Create Estimator Instance
# Create the t-SNE model.
# n_components: The number of dimensions to reduce to. Usually set to 2 or 3 for visualization.
# perplexity: A value indicating how many neighbors each data point should consider.
# The optimal value may vary depending on the data structure.
# n_iter: The number of iterations for optimization. Must be set large enough to ensure stable results.
# learning_rate: The learning rate. Too small can slow convergence; too large can lead to instability.
tsne = TSNE(n_components=2, perplexity=30, max_iter=300, random_state=42)

# 4. Model Training and Transformation (fit_transform)
# The model.fit_transform(X) method is used to train the model and transform the data.
# Since t-SNE is unsupervised learning, only the feature data (X) is used, without the correct labels (y).
print("Starting t-SNE dimensionality reduction...")
X_tsne = tsne.fit_transform(X_scaled)
print("t-SNE dimensionality reduction complete!")
print("\nOriginal data dimension:", X_scaled.shape)
print("Data dimension after t-SNE application:", X_tsne.shape)

# 5. Visualize Results
# Visualize the data reduced to 2 dimensions to identify the cluster structure.
# The color of each point represents the original digit label.
# Each point on the plot corresponds to one handwritten digit image that was in the 64-dimensional space.
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="Paired", s=30)
plt.title("t-SNE Visualization of Digits Dataset")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.colorbar(label="Digit Class")

# ==============================================================================
# Section 6: Comparison of PCA and t-SNE Results
# Apply PCA to the same data to compare how it differs from t-SNE.
# ==============================================================================
print("\nStarting PCA dimensionality reduction...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print("PCA dimensionality reduction complete!")

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
Execution Example:

Starting t-SNE dimensionality reduction...
t-SNE dimensionality reduction complete!

Original data dimension: (1797, 64)
Data dimension after t-SNE application: (1797, 2)
"""

"""
Review

Q1. Although t-SNE is a dimensionality reduction algorithm like PCA, what is its primary purpose? What is the biggest difference between the two algorithms?

    t-SNE is primarily used for **data visualization**.
    It excels at transforming the data into a low-dimensional space while preserving the **local (neighborhood) structure** between data points, making it easy to visually identify complex data cluster structures at a glance.

    Key Difference: PCA reduces dimensions by maximizing the data's variance, preserving the **global** structure.
    In contrast, t-SNE focuses on preserving the **local** structure, aiming to accurately represent the similarity between data points in the low-dimensional space.

Q2. What is the role of the perplexity parameter in the t-SNE model, and how does this value affect the result?

    **perplexity** is a value that defines **"how many neighbors a data point should consider"**.
    This value is important for finding a balance that considers both the local and global structure of the data.

    A small perplexity value: The algorithm focuses more on the local structure, which can cause small clusters to split into multiple groups.

    A large perplexity value: The algorithm considers more of the global structure, which can lead to the formation of larger clusters, and in complex data, might cause groups to appear overly clumped together.

Q3. Why is the data standardized with StandardScaler before applying t-SNE?

    t-SNE calculates similarity based on the **distance** between data points.
    If the scale of each feature is vastly different, the feature with the largest scale will disproportionately influence the distance calculation, potentially skewing the results.
    StandardScaler equalizes the scale of all features, ensuring that all features have equal importance in the distance calculation, which leads to a more accurate visualization.

Q4. It can be observed that the clusters are clearly separated in the t-SNE visualization result. What does this signify?

    This means that data points with the same label (e.g., images of the same digit) in the high-dimensional space are located close to each other,
    and far from data points with different labels.
    In other words, it demonstrates that t-SNE successfully preserved the complex relationships from the high-dimensional space in the low-dimensional representation.

Q5. Is visualization one of the main goals of dimensionality reduction?

    Yes, one of the primary goals of dimensionality reduction is **visualization**.

Q6. Is t-SNE better than PCA? Is it always better? Are there times when it is better? What are the selection criteria?

    t-SNE is **not always** better than PCA.
    t-SNE excels at visualizing cluster structure but is computationally expensive and may not be suitable for very large datasets.
    PCA is fast to compute and reduces dimensions by maximizing data variance, making it useful for understanding the overall data structure.
    The selection criteria depend on the analysis goal, data size, and computational resources.
    In this digit recognition example, t-SNE yielded better visualization results.

Q7. Should both PCA and t-SNE be attempted?

    Yes, PCA and t-SNE are complementary dimensionality reduction techniques, so it is generally recommended to try both.

Q8. Why the t-distribution?

    SNE uses the Gaussian distribution, but t-SNE uses the **t-distribution** to solve its **Crowding Problem**.
    The t-distribution has heavier tails than the Gaussian distribution, which allows it to better preserve the local structure of high-dimensional data.

Q9. Let's first understand the objective function of SNE.

    The objective function of SNE is to preserve the similarity between data points.
    SNE attempts to preserve the distances between data points in the high-dimensional space (64-dimensions in this example) as much as possible in the low-dimensional space (2-dimensions in this example).
    To do this, SNE probabilistically models the distance to each data point's neighbors and optimizes the arrangement to minimize the difference between these probability distributions.
    t-SNE is designed to improve SNE's objective function to better preserve the local structure.

    The core idea of SNE is to make the probability distribution in the high-dimensional space as similar as possible to the probability distribution in the low-dimensional space.
    The metric used to measure this "similarity" is the **Kullback-Leibler Divergence (KL Divergence)**.
    The objective function of SNE is as follows, and SNE adjusts the positions of the points to minimize the value C:

    $$C = \sum_{i} D_{KL}(P_i || Q_i) = \sum_{i} \sum_{j} P_{ij} \log \frac{P_{ij}}{Q_{ij}}$$

    Here, $P_{ij}$ is the similarity probability between data points $i$ and $j$ in the high-dimensional space (the probability that $x_i$ selects $x_j$ as a neighbor in the high-dimensional space), and $Q_{ij}$ is the similarity probability in the low-dimensional space.
    SNE minimizes this KL divergence to preserve the high-dimensional structure in the low-dimensional space.
    A KL divergence value close to 0 indicates that the two probability distributions are similar.

Q10. So, what is the Crowding Problem in SNE?

    The Crowding Problem arises because the Gaussian distribution is used to calculate the terms $P_{ij}$ and $Q_{ij}$ in the objective function.

    While points that are close in high dimensions being close in low dimensions is less of an issue, an error occurs when **points that are far apart in high dimensions end up being too close in low dimensions**.
    For example, two clusters, say for digit '1' and digit '7', might be far apart in high dimensions, but using SNE could cause them to merge in the low-dimensional space.
    This issue is called the **Crowding Problem**.

    t-SNE addressed this problem by using the **t-distribution** to ensure that the distances of points that were far apart in the high-dimensional space are appropriately maintained in the low-dimensional space as well.

    SNE was published in 2002.
    t-SNE was published in 2008.

Q11. What does it mean that SNE uses the Gaussian distribution?

    The SNE formula uses a shape similar to the PDF (Probability Density Function) of a 1D normal distribution.

    The PDF of a 1D normal distribution is:
    $$P(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}$$

    The exponential part of this, $e^{-\frac{(x - \mu)^2}{2\sigma^2}}$, is exactly similar to the way $P_{ij}$ is calculated in SNE's objective function.
    That part forms the bell curve shape of the normal distribution.
    The numerator in the SNE formula precisely follows the key bell curve of the Gaussian PDF.

    Therefore, SNE expresses the similarity between two points as a probability, borrowing the form of the Gaussian distribution.

Q12. The variance $\sigma^2$ of the normal distribution is included in the objective function. Is this value calculated because the input data is normally distributed, or is it calculated assuming that the distance values are normally distributed?

    The phrase "**calculated assuming a normal distribution**" is accurate.
    It is not a proven fact.

Q13. What is the best practice for selecting the parameters n_components=2 and perplexity=30?

    **n_components=2** is commonly used for visualization.
    Reducing to 2 dimensions allows for intuitive visualization of the data on a 2D plot.
    It can be reduced to 3 dimensions, but 2D is more common.

    **perplexity** depends on the data's density and structure.
    Generally, values between **5 and 50** are used.
    A small perplexity focuses more on local structure.
    A large perplexity focuses more on global structure.
    Therefore, it is important to find the optimal perplexity value based on the characteristics of the data.

    In general, perplexity is adjusted according to the data's size and density.
    For example, if the data is dense, a small perplexity value is better, and if the data is sparse, a large perplexity value is better.
    Using values between 5 and 50 is common, and finding the optimal value for the specific data is key.

Q14. Compare the difference between the objective functions of SNE and t-SNE.

    In SNE, the probability of two points being neighbors is calculated using:
        $P_{ij}$ in high dimensions
        $Q_{ij}$ in low dimensions
        **Both** use the **Gaussian distribution** formula.

    In t-SNE, the probability of two points being neighbors is calculated using:
        $P_{ij}$ in high dimensions is calculated using the **Gaussian distribution**, same as SNE.
        $Q_{ij}$ in low dimensions is calculated using the **t-distribution**, which has heavier tails.

    The shape of the objective function C is the same for both, but the formula used to calculate the similarity probability in the low-dimensional space (in the denominator) is replaced with the **t-distribution** shape in t-SNE.
"""
