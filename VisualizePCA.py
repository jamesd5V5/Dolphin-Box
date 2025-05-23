import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from DataExtraction import get_all_spectrograms_with_pca
from mpl_toolkits.mplot3d import Axes3D

data = get_all_spectrograms_with_pca()
specs = np.array([spec for spec, _ in data])
labels = np.array([label for _, label in data])

pca_10d = PCA(n_components=10)
reduced_data_10d = pca_10d.fit_transform(specs)

# Plot variance for each component
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(range(1, 11), pca_10d.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Variance Explained by Each Component')

# Cumuluative Variance
plt.subplot(1, 2, 2)
plt.plot(range(1, 11), np.cumsum(pca_10d.explained_variance_ratio_), 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Variance Explained')
plt.grid(True)
plt.tight_layout()
plt.savefig('pca_variance_10d.png')
plt.close()

print("\nVariance from component:")
for i, ratio in enumerate(pca_10d.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio:.2%}")

'''
print(f"\nCumulative variance: ")
for i in range(10):
    print(f"First {i+1} components: {sum(pca_10d.explained_variance_ratio_[:i+1]):.2%}")
    '''

plt.figure(figsize=(12, 8))
sns.heatmap(pca_10d.components_, 
            cmap='coolwarm',
            xticklabels=[f'PC{i+1}' for i in range(10)],
            yticklabels=[f'Feature {i+1}' for i in range(pca_10d.components_.shape[0])])
plt.title('PCA Component Weights')
plt.tight_layout()
plt.savefig('pca_components_heatmap.png')
plt.close()

# Plotted first 3 components for 4 classes
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
# Use a color map with 4 distinct colors
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
class_names = ['Whistle', 'Click', 'BP', 'Noise']
for class_idx, class_name in enumerate(class_names):
    mask = labels == class_idx
    ax.scatter(reduced_data_10d[mask, 0], reduced_data_10d[mask, 1], reduced_data_10d[mask, 2],
               c=colors[class_idx], label=class_name, alpha=0.6)
plt.colorbar(ax.scatter([], [], [], c=[]), label='Class (see legend)')
ax.set_xlabel(f'PC1 ({pca_10d.explained_variance_ratio_[0]:.2%} variance)')
ax.set_ylabel(f'PC2 ({pca_10d.explained_variance_ratio_[1]:.2%} variance)')
ax.set_zlabel(f'PC3 ({pca_10d.explained_variance_ratio_[2]:.2%} variance)')
plt.title('First 3 Principal Components (4 Classes)')
ax.legend()
plt.savefig('pca_3d_top3_4classes.png')
plt.close()

# 2x2 grid of 3D PCA plots: each class vs Noise, and all classes
fig = plt.figure(figsize=(18, 14))

class_names = ['Whistle', 'Click', 'BP', 'Noise']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# (1) Whistle vs Noise
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
for class_idx, class_name in [(0, 'Whistle'), (3, 'Noise')]:
    mask = labels == class_idx
    ax1.scatter(reduced_data_10d[mask, 0], reduced_data_10d[mask, 1], reduced_data_10d[mask, 2],
                c=colors[class_idx], label=class_name, alpha=0.6)
ax1.set_title('Whistle vs Noise')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_zlabel('PC3')
ax1.legend()

# (2) Click vs Noise
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
for class_idx, class_name in [(1, 'Click'), (3, 'Noise')]:
    mask = labels == class_idx
    ax2.scatter(reduced_data_10d[mask, 0], reduced_data_10d[mask, 1], reduced_data_10d[mask, 2],
                c=colors[class_idx], label=class_name, alpha=0.6)
ax2.set_title('Click vs Noise')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_zlabel('PC3')
ax2.legend()

# (3) BP vs Noise
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
for class_idx, class_name in [(2, 'BP'), (3, 'Noise')]:
    mask = labels == class_idx
    ax3.scatter(reduced_data_10d[mask, 0], reduced_data_10d[mask, 1], reduced_data_10d[mask, 2],
                c=colors[class_idx], label=class_name, alpha=0.6)
ax3.set_title('BP vs Noise')
ax3.set_xlabel('PC1')
ax3.set_ylabel('PC2')
ax3.set_zlabel('PC3')
ax3.legend()

# (4) All classes
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
for class_idx, class_name in enumerate(class_names):
    mask = labels == class_idx
    ax4.scatter(reduced_data_10d[mask, 0], reduced_data_10d[mask, 1], reduced_data_10d[mask, 2],
                c=colors[class_idx], label=class_name, alpha=0.6)
ax4.set_title('All Classes')
ax4.set_xlabel('PC1')
ax4.set_ylabel('PC2')
ax4.set_zlabel('PC3')
ax4.legend()

plt.tight_layout()
plt.savefig('pca_3d_grid_vs_noise.png')
plt.close()

def visualize_hidden_states(close_p, states, n_components, title="Time Series with Hidden States"):
    """
    Visualize time series data with hidden state assignments.
    
    Args:
        close_p: Array of closing prices
        states: Array of hidden state assignments
        n_components: Number of hidden states
        title: Main title for the plot
    """
    dates = np.arange(len(close_p))
    
    # Create figure with subplots
    fig, axs = plt.subplots(n_components + 1, 1, figsize=(12, 3*(n_components + 1)), sharex=True)
    fig.suptitle(title, fontsize=16)
    
    # Use a colormap for different states
    colours = plt.cm.rainbow(np.linspace(0, 1, n_components))
    
    # Plot raw data
    axs[0].plot(dates, close_p, 'k-', linewidth=1)
    axs[0].set_title("Raw Time Series Data")
    axs[0].grid(True)
    axs[0].set_ylabel("Price")
    
    # Plot data colored by state
    for i in range(n_components):
        mask = states == i
        axs[i+1].plot(dates[mask], close_p[mask], ".-", c=colours[i], linewidth=1)
        axs[i+1].set_title(f"State {i}")
        axs[i+1].grid(True)
        axs[i+1].set_ylabel("Price")
    
    # Set x-axis label only on the bottom subplot
    axs[-1].set_xlabel("Time")
    
    plt.tight_layout()
    return fig 