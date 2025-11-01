# ----------------------------------------------------------------------
# Helper Function for Visualization
# ----------------------------------------------------------------------
from matplotlib import pyplot as plt
from sklearn.tree import plot_tree


def visualize_decision_tree(model, feature_names, class_names):
    """
    Visualizes the Decision Tree structure using plot_tree.

    Args:
        model (DecisionTreeClassifier): The trained tree model instance.
        feature_names (list): List of feature names.
        class_names (list): List of class names.
    """
    # Set Matplotlib Figure size
    plt.figure(figsize=(25, 12))

    # Visualize the Decision Tree structure using plot_tree
    plot_tree(
        model,
        feature_names=feature_names,  # Feature names
        class_names=class_names,  # Class names
        filled=True,  # Color codes the classes
        rounded=True,  # Display node corners rounded
        proportion=True,  # Display proportions instead of sample counts
        fontsize=8,  # Adjust font size
    )

    # Use the model's max_depth attribute for the title
    plt.title(f"Decision Tree (Max Depth {model.max_depth})", fontsize=15)
    plt.show()
