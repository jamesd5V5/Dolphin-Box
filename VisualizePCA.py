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

#Plotted first 3 components
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(reduced_data_10d[:, 0], reduced_data_10d[:, 1], reduced_data_10d[:, 2],
                    c=labels, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Class')
ax.set_xlabel(f'PC1 ({pca_10d.explained_variance_ratio_[0]:.2%} variance)')
ax.set_ylabel(f'PC2 ({pca_10d.explained_variance_ratio_[1]:.2%} variance)')
ax.set_zlabel(f'PC3 ({pca_10d.explained_variance_ratio_[2]:.2%} variance)')
plt.title('First 3 Principal Components')
plt.savefig('pca_3d_top3.png')
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