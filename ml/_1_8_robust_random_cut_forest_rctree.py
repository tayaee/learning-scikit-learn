import numpy as np
import pandas as pd
import rrcf
import matplotlib.pyplot as plt

# -----------------
# 1. Parameter Settings
# -----------------
np.random.seed(42)
n_samples = 200  # Total number of data points
n_dimensions = 2  # Data dimension (set to 2D for visualization)
num_trees = 40  # Number of trees in the forest
tree_size = 32  # Number of data points each tree will randomly sample
anomaly_count = 5  # Number of intentional anomalies

# -----------------
# 2. Data Generation
# -----------------
# (A) Normal Data: A cluster centered around the origin
X_normal = np.random.normal(loc=0, scale=1, size=(n_samples - anomaly_count, n_dimensions))

# (B) Anomaly Data: 5 distant points
X_anomaly = np.array([[10, 10], [12, -10], [-10, 15], [20, 0], [0, 20]])

# Combine data
X = np.concatenate((X_normal, X_anomaly), axis=0)
df = pd.DataFrame(X, columns=["x", "y"])

# -----------------
# 3. Build RRCF Forest (using insert_point to resolve KeyError)
# -----------------
forest = []
while len(forest) < num_trees:
    # Randomly sample 'tree_size' from the entire data index for the ensemble
    # ixs are the indices of the original data (from 0 to 199)
    ixs = np.random.choice(len(X), size=tree_size, replace=False)

    # 1. Create an empty Random Cut Tree (RCTree)
    tree = rrcf.RCTree()

    # 2. Insert only the sampled data points into the tree (preserving the original index)
    for ix in ixs:
        # X[ix] is the point data, ix is the index of the original data
        tree.insert_point(X[ix], index=ix)

    forest.append(tree)

# -----------------
# 4. Calculate Anomaly Score (Co-Displacement: CoDisp)
# -----------------
# avg_codisp: Series to store the sum of CoDisp from all trees
# tree_count: Series to store the count of trees each point is included in
avg_codisp = pd.Series(0.0, index=df.index)
tree_count = pd.Series(0, index=df.index)

for tree in forest:
    # Calculate CoDisp only for the indices the tree knows about (tree.leaves)
    for index in tree.leaves:
        avg_codisp.loc[index] += tree.codisp(index)
        tree_count.loc[index] += 1

# Calculate the average (divide the sum of scores for each point by the number of trees that included that point)
# Handle division by zero (if a point was not sampled by any tree) by setting the result to 0
avg_codisp /= tree_count
avg_codisp.fillna(0, inplace=True)

# Add the score results to the DataFrame
df["Anomaly_Score"] = avg_codisp
df.sort_values(by="Anomaly_Score", ascending=False, inplace=True)

# -----------------
# 5. Visualize Results
# -----------------
# Consider the top 5 records with the highest anomaly scores as anomalies
top_anomalies = df.head(anomaly_count)

print("--- Top 5 Records with the Highest Anomaly Scores (Detected Anomalies) ---")
print(top_anomalies[["x", "y", "Anomaly_Score"]])
print("-" * 40)

plt.figure(figsize=(10, 7))

# Normal data points (low score data)
scatter = plt.scatter(
    df["x"],
    df["y"],
    c=df["Anomaly_Score"],
    cmap="viridis",
    s=50,
    label="Normal Data",
)

# Top 5 anomaly points (high score data)
plt.scatter(
    top_anomalies["x"],
    top_anomalies["y"],
    color="red",
    s=150,
    marker="X",
    label=f"Top {anomaly_count} Anomalies",
)

plt.colorbar(scatter, label="Anomaly Score (CoDisp)")
plt.title("Random Cut Forest")
plt.xlabel("X Feature")
plt.ylabel("Y Feature")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
