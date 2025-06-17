import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from DataExtraction import get_all_spectrograms_with_pca
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from collections import Counter
import umap
import pandas as pd
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

data = get_all_spectrograms_with_pca()
specs = np.array([spec for spec, _ in data])
labels = np.array([label for _, label in data])

pca_10d = PCA(n_components=10)
reduced_data_10d = pca_10d.fit_transform(specs)

reducer = umap.UMAP(n_components=3, random_state=42)
reduced_data_umap = reducer.fit_transform(specs)

class_names = ['Whistle', 'Click', 'BP', 'Noise']

pca_df_full = pd.DataFrame(reduced_data_10d, 
                          columns=[f'PC{i+1}' for i in range(10)])
pca_df_full['Class'] = [class_names[label] for label in labels]
umap_df = pd.DataFrame(reduced_data_umap, columns=['UMAP1', 'UMAP2', 'UMAP3'])
umap_df['Class'] = [class_names[label] for label in labels]\
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
parallel_coordinates(pca_df_full, 'Class', colormap=plt.cm.Set2)
plt.title('PCA: Parallel Coordinates Plot (All 10 Components)')
plt.xticks(rotation=45)
plt.grid(True)

plt.subplot(1, 2, 2)
parallel_coordinates(umap_df, 'Class', colormap=plt.cm.Set2)
plt.title('UMAP: Parallel Coordinates Plot')
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.savefig('parallel_coordinates.png')
plt.close()

pca_df_normalized = pca_df_full.copy()
umap_df_normalized = umap_df.copy()

for col in pca_df_normalized.columns:
    if col != 'Class':
        pca_df_normalized[col] = (pca_df_normalized[col] - pca_df_normalized[col].min()) / \
                                (pca_df_normalized[col].max() - pca_df_normalized[col].min())

for col in umap_df_normalized.columns:
    if col != 'Class':
        umap_df_normalized[col] = (umap_df_normalized[col] - umap_df_normalized[col].min()) / \
                                 (umap_df_normalized[col].max() - umap_df_normalized[col].min())

plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
parallel_coordinates(pca_df_normalized, 'Class', colormap=plt.cm.Set2)
plt.title('PCA: Normalized Parallel Coordinates Plot (All 10 Components)')
plt.xticks(rotation=45)
plt.grid(True)

plt.subplot(1, 2, 2)
parallel_coordinates(umap_df_normalized, 'Class', colormap=plt.cm.Set2)
plt.title('UMAP: Normalized Parallel Coordinates Plot')
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.savefig('parallel_coordinates_normalized.png')
plt.close()

for class_name in class_names:
    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 2, 1)
    class_data = pca_df_normalized[pca_df_normalized['Class'] == class_name]
    parallel_coordinates(class_data, 'Class', colormap=plt.cm.Set2)
    plt.title(f'PCA: Normalized Parallel Coordinates Plot for {class_name}')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    class_data = umap_df_normalized[umap_df_normalized['Class'] == class_name]
    parallel_coordinates(class_data, 'Class', colormap=plt.cm.Set2)
    plt.title(f'UMAP: Normalized Parallel Coordinates Plot for {class_name}')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'parallel_coordinates_{class_name.lower()}.png')
    plt.close()

plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
for class_name in class_names:
    class_data = pca_df_full[pca_df_full['Class'] == class_name]
    plt.scatter(class_data['PC1'], class_data['PC2'], alpha=0.5, label=class_name)
plt.title('PCA: PC1 vs PC2 Density')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()

plt.subplot(2, 2, 2)
for class_name in class_names:
    class_data = pca_df_full[pca_df_full['Class'] == class_name]
    plt.scatter(class_data['PC1'], class_data['PC3'], alpha=0.5, label=class_name)
plt.title('PCA: PC1 vs PC3 Density')
plt.xlabel('PC1')
plt.ylabel('PC3')
plt.legend()

plt.subplot(2, 2, 3)
for class_name in class_names:
    class_data = umap_df[umap_df['Class'] == class_name]
    plt.scatter(class_data['UMAP1'], class_data['UMAP2'], alpha=0.5, label=class_name)
plt.title('UMAP: UMAP1 vs UMAP2 Density')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.legend()

plt.subplot(2, 2, 4)
for class_name in class_names:
    class_data = umap_df[umap_df['Class'] == class_name]
    plt.scatter(class_data['UMAP1'], class_data['UMAP3'], alpha=0.5, label=class_name)
plt.title('UMAP: UMAP1 vs UMAP3 Density')
plt.xlabel('UMAP1')
plt.ylabel('UMAP3')
plt.legend()

plt.tight_layout()
plt.savefig('density_plots.png')
plt.close()

scaler_pca = StandardScaler()
scaler_umap = StandardScaler()

pca_df_zscore = pca_df_full.copy()
umap_df_zscore = umap_df.copy()

pca_features = [col for col in pca_df_zscore.columns if col != 'Class']
umap_features = [col for col in umap_df_zscore.columns if col != 'Class']

pca_df_zscore[pca_features] = scaler_pca.fit_transform(pca_df_zscore[pca_features])
umap_df_zscore[umap_features] = scaler_umap.fit_transform(umap_df_zscore[umap_features])

class_color_dict = {
    'Whistle': '#1f77b4',  # blue
    'Click':   '#ff7f0e',  # orange
    'BP':      '#2ca02c',  # green
    'Noise':   '#7f7f7f',  # gray
}
class_order = ['Whistle', 'Click', 'BP', 'Noise']
class_colors = [class_color_dict[c] for c in class_order]

z_score_threshold = 3.0
noise_threshold = 2.0
pca_df_zscore_no_outliers = pca_df_zscore.copy()
umap_df_zscore_no_outliers = umap_df_zscore.copy()

def remove_outliers_class_wise(df, features, threshold, noise_threshold):
    mask = np.ones(len(df), dtype=bool)
    for class_name in df['Class'].unique():
        class_mask = df['Class'] == class_name
        class_data = df[class_mask]
        current_threshold = noise_threshold if class_name == 'Noise' else threshold
        for feature in features:
            class_mean = class_data[feature].mean()
            class_std = class_data[feature].std()
            
            class_feature_mask = np.abs((class_data[feature] - class_mean) / class_std) <= current_threshold
            mask[class_mask] &= class_feature_mask
    
    return df[mask]

pca_df_zscore_no_outliers = remove_outliers_class_wise(pca_df_zscore_no_outliers, pca_features, z_score_threshold, noise_threshold)
umap_df_zscore_no_outliers = remove_outliers_class_wise(umap_df_zscore_no_outliers, umap_features, z_score_threshold, noise_threshold)

print("Original data size:", len(pca_df_zscore))
print("Data size after class-wise outlier removal:", len(pca_df_zscore_no_outliers))
print("Number of points removed:", len(pca_df_zscore) - len(pca_df_zscore_no_outliers))
print("\nPoints removed per class:")
for class_name in pca_df_zscore['Class'].unique():
    original_count = len(pca_df_zscore[pca_df_zscore['Class'] == class_name])
    remaining_count = len(pca_df_zscore_no_outliers[pca_df_zscore_no_outliers['Class'] == class_name])
    removed_count = original_count - remaining_count
    threshold_used = noise_threshold if class_name == 'Noise' else z_score_threshold
    print(f"{class_name}: {removed_count} points removed ({removed_count/original_count:.1%} of class) [threshold: {threshold_used}]")

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
sns.boxplot(data=pca_df_zscore_no_outliers, x='Class', y='PC1', hue='Class', 
            palette=class_color_dict, legend=False)
plt.title('PC1 Distribution by Class\n(After Outlier Removal)')
plt.xticks(rotation=45)

plt.subplot(2, 3, 2)
sns.boxplot(data=pca_df_zscore_no_outliers, x='Class', y='PC2', hue='Class',
            palette=class_color_dict, legend=False)
plt.title('PC2 Distribution by Class\n(After Outlier Removal)')
plt.xticks(rotation=45)

plt.subplot(2, 3, 3)
sns.boxplot(data=pca_df_zscore_no_outliers, x='Class', y='PC3', hue='Class',
            palette=class_color_dict, legend=False)
plt.title('PC3 Distribution by Class\n(After Outlier Removal)')
plt.xticks(rotation=45)

plt.subplot(2, 3, 4)
sns.boxplot(data=umap_df_zscore_no_outliers, x='Class', y='UMAP1', hue='Class',
            palette=class_color_dict, legend=False)
plt.title('UMAP1 Distribution by Class\n(After Outlier Removal)')
plt.xticks(rotation=45)

plt.subplot(2, 3, 5)
sns.boxplot(data=umap_df_zscore_no_outliers, x='Class', y='UMAP2', hue='Class',
            palette=class_color_dict, legend=False)
plt.title('UMAP2 Distribution by Class\n(After Outlier Removal)')
plt.xticks(rotation=45)

plt.subplot(2, 3, 6)
sns.boxplot(data=umap_df_zscore_no_outliers, x='Class', y='UMAP3', hue='Class',
            palette=class_color_dict, legend=False)
plt.title('UMAP3 Distribution by Class\n(After Outlier Removal)')
plt.xticks(rotation=45)

plt.suptitle('Distribution of Components by Class\n(After Class-wise Outlier Removal, Noise threshold: 2.0σ)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('distribution_boxplots_cleaned.png')
plt.close()

plt.figure(figsize=(15, 10))
selected_components = ['PC3', 'PC5', 'PC6', 'PC8', 'PC10']

for idx, component in enumerate(selected_components, 1):
    plt.subplot(2, 3, idx)
    sns.boxplot(data=pca_df_zscore_no_outliers, x='Class', y=component, hue='Class',
                palette=class_color_dict, legend=False)
    plt.title(f'{component} Distribution by Class\n(After Outlier Removal)')
    plt.xticks(rotation=45)

plt.suptitle('Distribution of Selected Components by Class\n(After Class-wise Outlier Removal, Noise threshold: 2.0σ)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('selected_components_boxplots_cleaned.png')
plt.close()

print("\nStatistics for selected components after outlier removal:")
for component in selected_components:
    print(f"\n{component}:")
    for class_name in class_names:
        class_data = pca_df_zscore_no_outliers[pca_df_zscore_no_outliers['Class'] == class_name][component]
        print(f"{class_name}:")
        print(f"  Mean: {class_data.mean():.3f}")
        print(f"  Std: {class_data.std():.3f}")
        print(f"  Min: {class_data.min():.3f}")
        print(f"  Max: {class_data.max():.3f}")
        print(f"  Count: {len(class_data)}")

# Print correlation matrices
print("\n=== PCA Correlation Matrix ===")
print(pca_df_full[['PC1', 'PC2', 'PC3']].corr())

print("\n=== UMAP Correlation Matrix ===")
print(umap_df[['UMAP1', 'UMAP2', 'UMAP3']].corr())

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(range(1, 11), pca_10d.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Variance Explained by Each Component')

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
plt.savefig('NEWER_pca_3d_top3_4classes.png')
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
plt.savefig('Newsest_pca_3d_grid_vs_noise.png')
plt.close()

# K-means clustering on the first 3 PCA components
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(reduced_data_10d[:, :3])

# Plot k-means clusters in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
cluster_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for cluster_idx in range(4):
    mask = kmeans_labels == cluster_idx
    ax.scatter(reduced_data_10d[mask, 0], reduced_data_10d[mask, 1], reduced_data_10d[mask, 2],
               c=cluster_colors[cluster_idx], label=f'Cluster {cluster_idx}', alpha=0.6)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('K-means Clusters (k=4) in PCA Space')
ax.legend()
plt.tight_layout()
plt.savefig('pca_3d_kmeans_clusters.png')
plt.close()

# Print cluster-to-class mapping and cluster sizes
print('\nK-means cluster sizes:')
for i in range(4):
    print(f'Cluster {i}: {np.sum(kmeans_labels == i)} samples')

print('\nCluster-to-class mapping (majority vote):')
for i in range(4):
    true_labels = labels[kmeans_labels == i]
    if len(true_labels) > 0:
        most_common = Counter(true_labels).most_common(1)[0]
        print(f'Cluster {i}: Most common true class = {most_common[0]} (count={most_common[1]})')
    else:
        print(f'Cluster {i}: No samples')

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

# Plot UMAP 3D visualization
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
for class_idx, class_name in enumerate(class_names):
    mask = labels == class_idx
    ax.scatter(reduced_data_umap[mask, 0], reduced_data_umap[mask, 1], reduced_data_umap[mask, 2],
               c=colors[class_idx], label=class_name, alpha=0.6)
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.set_zlabel('UMAP3')
plt.title('UMAP: 3D Projection')
ax.legend()
plt.savefig('umap_3d_visualization.png')
plt.close()

# Compare PCA and UMAP in 2D
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# PCA 2D
for class_idx, class_name in enumerate(class_names):
    mask = labels == class_idx
    ax1.scatter(reduced_data_10d[mask, 0], reduced_data_10d[mask, 1],
                c=colors[class_idx], label=class_name, alpha=0.6)
ax1.set_xlabel(f'PC1 ({pca_10d.explained_variance_ratio_[0]:.2%} variance)')
ax1.set_ylabel(f'PC2 ({pca_10d.explained_variance_ratio_[1]:.2%} variance)')
ax1.set_title('PCA: First 2 Principal Components')
ax1.legend()

# UMAP 2D
reducer_2d = umap.UMAP(n_components=2, random_state=42)
reduced_data_umap_2d = reducer_2d.fit_transform(specs)
for class_idx, class_name in enumerate(class_names):
    mask = labels == class_idx
    ax2.scatter(reduced_data_umap_2d[mask, 0], reduced_data_umap_2d[mask, 1],
                c=colors[class_idx], label=class_name, alpha=0.6)
ax2.set_xlabel('UMAP1')
ax2.set_ylabel('UMAP2')
ax2.set_title('UMAP: 2D Projection')
ax2.legend()

plt.tight_layout()
plt.savefig('pca_vs_umap_2d.png')
plt.close()

# K-means clustering on both PCA and UMAP
kmeans_pca = KMeans(n_clusters=4, random_state=42)
kmeans_umap = KMeans(n_clusters=4, random_state=42)

kmeans_labels_pca = kmeans_pca.fit_predict(reduced_data_10d[:, :3])
kmeans_labels_umap = kmeans_umap.fit_predict(reduced_data_umap)

# Plot k-means clusters
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), subplot_kw={'projection': '3d'})

# PCA clusters
for cluster_idx in range(4):
    mask = kmeans_labels_pca == cluster_idx
    ax1.scatter(reduced_data_10d[mask, 0], reduced_data_10d[mask, 1], reduced_data_10d[mask, 2],
                c=colors[cluster_idx], label=f'Cluster {cluster_idx}', alpha=0.6)
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_zlabel('PC3')
ax1.set_title('K-means Clusters in PCA Space')
ax1.legend()

# UMAP clusters
for cluster_idx in range(4):
    mask = kmeans_labels_umap == cluster_idx
    ax2.scatter(reduced_data_umap[mask, 0], reduced_data_umap[mask, 1], reduced_data_umap[mask, 2],
                c=colors[cluster_idx], label=f'Cluster {cluster_idx}', alpha=0.6)
ax2.set_xlabel('UMAP1')
ax2.set_ylabel('UMAP2')
ax2.set_zlabel('UMAP3')
ax2.set_title('K-means Clusters in UMAP Space')
ax2.legend()

plt.tight_layout()
plt.savefig('kmeans_clusters_comparison.png')
plt.close()

# Print cluster statistics
print('\nPCA K-means cluster sizes:')
for i in range(4):
    print(f'Cluster {i}: {np.sum(kmeans_labels_pca == i)} samples')

print('\nUMAP K-means cluster sizes:')
for i in range(4):
    print(f'Cluster {i}: {np.sum(kmeans_labels_umap == i)} samples')

print('\nPCA Cluster-to-class mapping (majority vote):')
for i in range(4):
    true_labels = labels[kmeans_labels_pca == i]
    if len(true_labels) > 0:
        most_common = Counter(true_labels).most_common(1)[0]
        print(f'Cluster {i}: Most common true class = {most_common[0]} (count={most_common[1]})')
    else:
        print(f'Cluster {i}: No samples')

print('\nUMAP Cluster-to-class mapping (majority vote):')
for i in range(4):
    true_labels = labels[kmeans_labels_umap == i]
    if len(true_labels) > 0:
        most_common = Counter(true_labels).most_common(1)[0]
        print(f'Cluster {i}: Most common true class = {most_common[0]} (count={most_common[1]})')
    else:
        print(f'Cluster {i}: No samples')

# Create a grid PNG for z-score normalized parallel coordinates with fixed colors (no outliers)
fig, axs = plt.subplots(2, 2, figsize=(24, 16))
grid_classes = ['Whistle', 'Click', 'BP']

# Calculate global min and max for consistent y-axis
plot_cols = [col for col in pca_df_zscore_no_outliers.columns if col.startswith('PC')]
global_min = pca_df_zscore_no_outliers[plot_cols].min().min()
global_max = pca_df_zscore_no_outliers[plot_cols].max().max()
# Add some padding to the range
y_padding = (global_max - global_min) * 0.1
y_min = global_min - y_padding
y_max = global_max + y_padding

for idx, class_name in enumerate(grid_classes):
    ax = axs[idx // 2, idx % 2]
    class_data = pca_df_zscore_no_outliers[pca_df_zscore_no_outliers['Class'] == class_name].copy()
    # Only keep feature columns and 'Class'
    plot_cols = [col for col in class_data.columns if col.startswith('PC')] + ['Class']
    parallel_coordinates(class_data[plot_cols], 'Class', color=[class_color_dict[class_name]], ax=ax)
    ax.set_title(f'Z-score Normalized Parallel Coordinates for {class_name}', fontsize=16)
    ax.set_xticklabels(plot_cols[:-1], rotation=45)
    ax.grid(True)
    # Set consistent y-axis limits
    ax.set_ylim(y_min, y_max)

# Last panel: all classes including Noise
ax = axs[1, 1]
plot_cols = [col for col in pca_df_zscore_no_outliers.columns if col.startswith('PC')] + ['Class']
parallel_coordinates(pca_df_zscore_no_outliers[plot_cols], 'Class', color=class_colors, ax=ax)
ax.set_title('Z-score Normalized Parallel Coordinates for All Classes', fontsize=16)
ax.set_xticklabels(plot_cols[:-1], rotation=45)
ax.grid(True)
# Set consistent y-axis limits
ax.set_ylim(y_min, y_max)

plt.tight_layout()
plt.savefig('parallel_coordinates_zscore_grid_colored_no_outliers_class_wise.png')
plt.close()

# Create a comparison visualization showing all classes together
plt.figure(figsize=(20, 10))
plot_cols = [col for col in pca_df_zscore_no_outliers.columns if col.startswith('PC')] + ['Class']
parallel_coordinates(pca_df_zscore_no_outliers[plot_cols], 'Class', color=class_colors)
plt.title('PCA: Z-score Normalized Parallel Coordinates for All Classes\n(Class-wise Outlier Removal)', fontsize=16)
plt.xticks(rotation=45)
plt.grid(True)
# Set consistent y-axis limits
plt.ylim(y_min, y_max)
plt.tight_layout()
plt.savefig('parallel_coordinates_zscore_all_classes_class_wise.png')
plt.close()

# Fix the error in the Click vs BP visualization
# Create visualization for specific components comparing Clicks and BPs
selected_components = ['PC3', 'PC5', 'PC6', 'PC8', 'PC10']
click_bp_data = pca_df_zscore_no_outliers[pca_df_zscore_no_outliers['Class'].isin(['Click', 'BP'])].copy()

# Get variance information for selected components
component_variance = {f'PC{i+1}': ratio for i, ratio in enumerate(pca_10d.explained_variance_ratio_) 
                     if f'PC{i+1}' in selected_components}

# Create parallel coordinates plot for selected components with enhanced visualization
plt.figure(figsize=(15, 8))
# Add variance information to component labels
component_labels = [f'{comp}\n({component_variance[comp]:.1%} variance)' for comp in selected_components]
click_bp_data_plot = click_bp_data[selected_components + ['Class']].copy()
click_bp_data_plot.columns = component_labels + ['Class']

# Create the parallel coordinates plot with enhanced styling
parallel_coordinates(click_bp_data_plot, 'Class', 
                    color=[class_color_dict['Click'], class_color_dict['BP']],
                    alpha=0.7,
                    linewidth=1.5)

plt.title('PCA: Selected Components for Click vs BP Comparison\n' + 
          'Components: PC3, PC5, PC6, PC8, PC10', fontsize=14, pad=20)
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# Add legend with custom styling
legend_elements = [plt.Line2D([0], [0], color=class_color_dict['Click'], label='Click', linewidth=2),
                  plt.Line2D([0], [0], color=class_color_dict['BP'], label='BP', linewidth=2)]
plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

# Add variance information in the title
variance_info = '\n'.join([f'{comp}: {var:.1%} variance' for comp, var in component_variance.items()])
plt.figtext(0.02, 0.02, f'Component Variance:\n{variance_info}', 
            fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

# Set consistent y-axis limits
plt.ylim(y_min, y_max)

plt.tight_layout()
plt.savefig('click_bp_selected_components.png', bbox_inches='tight', dpi=300)
plt.close()

# Create individual box plots for each selected component with enhanced visualization
plt.figure(figsize=(15, 10))
for idx, component in enumerate(selected_components, 1):
    plt.subplot(2, 3, idx)
    sns.boxplot(x='Class', y=component, data=click_bp_data, 
                palette=[class_color_dict['Click'], class_color_dict['BP']])
    plt.title(f'{component}\n({component_variance[component]:.1%} variance)')
    plt.xticks(rotation=45)
    
    # Add statistical information
    click_data = click_bp_data[click_bp_data['Class'] == 'Click'][component]
    bp_data = click_bp_data[click_bp_data['Class'] == 'BP'][component]
    
    # Calculate and display mean difference
    mean_diff = abs(click_data.mean() - bp_data.mean())
    plt.text(0.5, plt.ylim()[1], f'Mean diff: {mean_diff:.2f}', 
             ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.8))

plt.suptitle('Distribution of Selected Components for Click vs BP Comparison', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('click_bp_selected_components_boxplots.png', bbox_inches='tight', dpi=300)
plt.close()

# Create scatter plot matrix for selected components with enhanced visualization
plt.figure(figsize=(20, 20))
for i, comp1 in enumerate(selected_components):
    for j, comp2 in enumerate(selected_components):
        if i != j:
            plt.subplot(len(selected_components), len(selected_components), i*len(selected_components) + j + 1)
            for class_name in ['Click', 'BP']:
                class_data = click_bp_data[click_bp_data['Class'] == class_name]
                plt.scatter(class_data[comp1], class_data[comp2], 
                          c=[class_color_dict[class_name]], 
                          label=class_name, alpha=0.6)
            
            # Add component labels with variance information
            if i == len(selected_components)-1:
                plt.xlabel(f'{comp1}\n({component_variance[comp1]:.1%} variance)')
            if j == 0:
                plt.ylabel(f'{comp2}\n({component_variance[comp2]:.1%} variance)')
            
            # Add legend only to the first subplot
            if i == 0 and j == 1:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Add grid
            plt.grid(True, alpha=0.3)

plt.suptitle('Scatter Plot Matrix of Selected Components for Click vs BP Comparison', fontsize=16, y=0.95)
plt.tight_layout()
plt.savefig('click_bp_selected_components_scatter_matrix.png', bbox_inches='tight', dpi=300)
plt.close()

# Create detailed box plots for each PCA component to analyze variance
plt.figure(figsize=(20, 15))

# Get all PCA components
pca_components = [col for col in pca_df_zscore_no_outliers.columns if col.startswith('PC')]

# Create box plots for each component
for idx, component in enumerate(pca_components, 1):
    plt.subplot(4, 3, idx)
    sns.boxplot(data=pca_df_zscore_no_outliers, x='Class', y=component, hue='Class',
                palette=class_color_dict, legend=False)
    
    # Add variance information to title
    class_variances = pca_df_zscore_no_outliers.groupby('Class')[component].var()
    variance_info = '\n'.join([f'{cls}: {var:.3f}' for cls, var in class_variances.items()])
    plt.title(f'{component} Distribution\nVariance by Class:\n{variance_info}')
    plt.xticks(rotation=45)
    
    # Add mean values as text
    class_means = pca_df_zscore_no_outliers.groupby('Class')[component].mean()
    for i, (cls, mean) in enumerate(class_means.items()):
        plt.text(i, plt.ylim()[0], f'μ={mean:.2f}', 
                ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.8))

plt.suptitle('Component-wise Distribution Analysis\n(After Class-wise Outlier Removal)', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('component_variance_analysis.png', bbox_inches='tight', dpi=300)
plt.close()

# Create a heatmap of variances to better visualize the differences
variance_matrix = pd.DataFrame(index=class_names, columns=pca_components)
for component in pca_components:
    for class_name in class_names:
        class_data = pca_df_zscore_no_outliers[pca_df_zscore_no_outliers['Class'] == class_name][component]
        variance_matrix.loc[class_name, component] = class_data.var()

# Convert variance matrix to float type
variance_matrix = variance_matrix.astype(float)

plt.figure(figsize=(15, 8))
sns.heatmap(variance_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
            cbar_kws={'label': 'Variance'})
plt.title('Variance by Component and Class\n(After Class-wise Outlier Removal)')
plt.tight_layout()
plt.savefig('variance_heatmap.png', bbox_inches='tight', dpi=300)
plt.close()

# Print detailed variance statistics
print("\nDetailed variance analysis by component and class:")
for component in pca_components:
    print(f"\n{component}:")
    for class_name in class_names:
        class_data = pca_df_zscore_no_outliers[pca_df_zscore_no_outliers['Class'] == class_name][component]
        print(f"{class_name}:")
        print(f"  Variance: {class_data.var():.3f}")
        print(f"  Std Dev: {class_data.std():.3f}")
        print(f"  Range: {class_data.max() - class_data.min():.3f}")
        print(f"  IQR: {class_data.quantile(0.75) - class_data.quantile(0.25):.3f}")
        print(f"  Count: {len(class_data)}")

# Create parallel coordinates plot with variance information
plt.figure(figsize=(20, 10))
plot_cols = [col for col in pca_df_zscore_no_outliers.columns if col.startswith('PC')] + ['Class']

# Calculate variance for each class and component
variance_info = {}
for class_name in class_names:
    class_data = pca_df_zscore_no_outliers[pca_df_zscore_no_outliers['Class'] == class_name]
    variance_info[class_name] = {col: class_data[col].var() for col in plot_cols[:-1]}

# Create parallel coordinates plot
parallel_coordinates(pca_df_zscore_no_outliers[plot_cols], 'Class', color=class_colors)

# Add variance information as text
for i, class_name in enumerate(class_names):
    class_data = pca_df_zscore_no_outliers[pca_df_zscore_no_outliers['Class'] == class_name]
    avg_var = np.mean([variance_info[class_name][col] for col in plot_cols[:-1]])
    max_var = max(variance_info[class_name].values())
    min_var = min(variance_info[class_name].values())
    plt.text(0.02, 0.95 - i*0.05, 
             f'{class_name}:\n  Avg var: {avg_var:.3f}\n  Max var: {max_var:.3f}\n  Min var: {min_var:.3f}',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

plt.title('PCA: Z-score Normalized Parallel Coordinates\n' + 
          'Variance information shown in legend', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('parallel_coordinates_variance_info.png', bbox_inches='tight', dpi=300)
plt.close()

# Create parallel coordinates plot with separate panels for high/low variance components
# Calculate component-wise variance differences
component_variance_diffs = {}
for component in pca_components:
    class_variances = [pca_df_zscore_no_outliers[pca_df_zscore_no_outliers['Class'] == cls][component].var() 
                      for cls in class_names]
    component_variance_diffs[component] = np.std(class_variances)

# Sort components by variance difference
sorted_components = sorted(component_variance_diffs.items(), key=lambda x: x[1], reverse=True)
high_var_components = [comp for comp, diff in sorted_components[:5]]  # Top 5 most variable components
low_var_components = [comp for comp, diff in sorted_components[5:]]  # Remaining components

# Create a figure with three panels: high variance, low variance, and class-specific patterns
fig = plt.figure(figsize=(20, 12))

# High variance components
ax1 = plt.subplot(2, 2, 1)
plot_cols_high = high_var_components + ['Class']
parallel_coordinates(pca_df_zscore_no_outliers[plot_cols_high], 'Class', 
                    color=class_colors, ax=ax1)
ax1.set_title('High Variance Components\n' + 
              '\n'.join([f'{comp}: {component_variance_diffs[comp]:.3f}' for comp in high_var_components]))
ax1.set_xticklabels(high_var_components, rotation=45)
ax1.grid(True)

# Low variance components
ax2 = plt.subplot(2, 2, 2)
plot_cols_low = low_var_components + ['Class']
parallel_coordinates(pca_df_zscore_no_outliers[plot_cols_low], 'Class', 
                    color=class_colors, ax=ax2)
ax2.set_title('Low Variance Components\n' + 
              '\n'.join([f'{comp}: {component_variance_diffs[comp]:.3f}' for comp in low_var_components]))
ax2.set_xticklabels(low_var_components, rotation=45)
ax2.grid(True)

# Class-specific patterns
ax3 = plt.subplot(2, 2, (3, 4))
# Find components where each class has highest variance
class_max_var_components = {}
for class_name in class_names:
    class_variances = {comp: variance_info[class_name][comp] for comp in pca_components}
    max_var_comp = max(class_variances.items(), key=lambda x: x[1])[0]
    class_max_var_components[class_name] = max_var_comp

# Create parallel coordinates plot for class-specific components
plot_cols_class = list(set(class_max_var_components.values())) + ['Class']
parallel_coordinates(pca_df_zscore_no_outliers[plot_cols_class], 'Class', 
                    color=class_colors, ax=ax3)
ax3.set_title('Class-Specific High Variance Components\n' + 
              '\n'.join([f'{cls}: {comp} ({variance_info[cls][comp]:.3f})' 
                        for cls, comp in class_max_var_components.items()]))
ax3.set_xticklabels(plot_cols_class[:-1], rotation=45)
ax3.grid(True)

plt.suptitle('Parallel Coordinates Analysis by Variance Patterns\n(After Class-wise Outlier Removal)', fontsize=14)
plt.tight_layout()
plt.savefig('parallel_coordinates_variance_analysis.png', bbox_inches='tight', dpi=300)
plt.close()

# Create UMAP and t-SNE visualizations without Noise class
# First, create a variance-weighted version of the data (using all data for weights)
weighted_data = pca_df_zscore_no_outliers.copy()

# Get PCA explained variance ratios
explained_variances = pca_10d.explained_variance_ratio_

# Calculate class-wise weights for each component (using all data)
for i, component in enumerate(pca_components):
    # Combine PCA explained variance with class-wise variance
    class_variances = []
    for class_name in class_names:
        class_data = weighted_data[weighted_data['Class'] == class_name][component]
        class_variances.append(class_data.var())
    
    # Calculate weights using both PCA explained variance and class-wise variance
    mean_class_variance = np.mean(class_variances)
    pca_weight = explained_variances[i]
    weight = (1.0 + mean_class_variance) * pca_weight  # Higher variance and higher PCA importance = higher weight
    
    # Apply weight to the component
    weighted_data[component] = weighted_data[component] * weight

# Verify no NaN values
assert not weighted_data[pca_components].isna().any().any(), "NaN values found in weighted data"

# Now filter out Noise class after weighting
main_classes_data = weighted_data[weighted_data['Class'] != 'Noise'].copy()
main_class_names = ['Whistle', 'Click', 'BP']

# Try different UMAP parameters without Noise
# 1. Standard UMAP
reducer1_no_noise = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
weighted_umap1_no_noise = reducer1_no_noise.fit_transform(main_classes_data[pca_components])

# 2. UMAP with more emphasis on local structure
reducer2_no_noise = umap.UMAP(n_components=2, random_state=42, n_neighbors=5, min_dist=0.3)
weighted_umap2_no_noise = reducer2_no_noise.fit_transform(main_classes_data[pca_components])

# 3. UMAP with more emphasis on global structure
reducer3_no_noise = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.01)
weighted_umap3_no_noise = reducer3_no_noise.fit_transform(main_classes_data[pca_components])

# Create UMAP visualizations without Noise
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Plot 1: Standard UMAP
for i, class_name in enumerate(main_class_names):
    mask = main_classes_data['Class'] == class_name
    axes[0].scatter(weighted_umap1_no_noise[mask, 0], weighted_umap1_no_noise[mask, 1], 
                   c=[class_color_dict[class_name]], label=class_name, alpha=0.6)
axes[0].set_title('Standard UMAP\n(n_neighbors=15, min_dist=0.1)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Local structure UMAP
for i, class_name in enumerate(main_class_names):
    mask = main_classes_data['Class'] == class_name
    axes[1].scatter(weighted_umap2_no_noise[mask, 0], weighted_umap2_no_noise[mask, 1], 
                   c=[class_color_dict[class_name]], label=class_name, alpha=0.6)
axes[1].set_title('Local Structure UMAP\n(n_neighbors=5, min_dist=0.3)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Global structure UMAP
for i, class_name in enumerate(main_class_names):
    mask = main_classes_data['Class'] == class_name
    axes[2].scatter(weighted_umap3_no_noise[mask, 0], weighted_umap3_no_noise[mask, 1], 
                   c=[class_color_dict[class_name]], label=class_name, alpha=0.6)
axes[2].set_title('Global Structure UMAP\n(n_neighbors=30, min_dist=0.01)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('UMAP Visualization\n(Weighted by Component Variance)', fontsize=14, y=1.05)
plt.tight_layout()
plt.savefig('umap_comparison_no_noise.png', bbox_inches='tight', dpi=300)
plt.close()

# Try different t-SNE parameters without Noise
# 1. Standard t-SNE
tsne1_no_noise = TSNE(n_components=2, random_state=42, perplexity=30)
weighted_tsne1_no_noise = tsne1_no_noise.fit_transform(main_classes_data[pca_components])

# 2. t-SNE with higher perplexity
tsne2_no_noise = TSNE(n_components=2, random_state=42, perplexity=50)
weighted_tsne2_no_noise = tsne2_no_noise.fit_transform(main_classes_data[pca_components])

# 3. t-SNE with lower perplexity
tsne3_no_noise = TSNE(n_components=2, random_state=42, perplexity=15)
weighted_tsne3_no_noise = tsne3_no_noise.fit_transform(main_classes_data[pca_components])

# Create t-SNE visualizations without Noise
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Plot 1: Standard t-SNE
for i, class_name in enumerate(main_class_names):
    mask = main_classes_data['Class'] == class_name
    axes[0].scatter(weighted_tsne1_no_noise[mask, 0], weighted_tsne1_no_noise[mask, 1], 
                   c=[class_color_dict[class_name]], label=class_name, alpha=0.6)
axes[0].set_title('Standard t-SNE\n(perplexity=30)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: High perplexity t-SNE
for i, class_name in enumerate(main_class_names):
    mask = main_classes_data['Class'] == class_name
    axes[1].scatter(weighted_tsne2_no_noise[mask, 0], weighted_tsne2_no_noise[mask, 1], 
                   c=[class_color_dict[class_name]], label=class_name, alpha=0.6)
axes[1].set_title('High Perplexity t-SNE\n(perplexity=50)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Low perplexity t-SNE
for i, class_name in enumerate(main_class_names):
    mask = main_classes_data['Class'] == class_name
    axes[2].scatter(weighted_tsne3_no_noise[mask, 0], weighted_tsne3_no_noise[mask, 1], 
                   c=[class_color_dict[class_name]], label=class_name, alpha=0.6)
axes[2].set_title('Low Perplexity t-SNE\n(perplexity=15)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('t-SNE Visualization\n(Weighted by Component Variance)', fontsize=14, y=1.05)
plt.tight_layout()
plt.savefig('tsne_comparison_no_noise.png', bbox_inches='tight', dpi=300)
plt.close()

# Create 3D UMAP visualization without Noise
reducer_3d_no_noise = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
weighted_umap_3d_no_noise = reducer_3d_no_noise.fit_transform(main_classes_data[pca_components])

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each class
for class_name in main_class_names:
    mask = main_classes_data['Class'] == class_name
    ax.scatter(weighted_umap_3d_no_noise[mask, 0], 
              weighted_umap_3d_no_noise[mask, 1], 
              weighted_umap_3d_no_noise[mask, 2],
              c=[class_color_dict[class_name]], 
              label=class_name, 
              alpha=0.6)

ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.set_zlabel('UMAP3')
plt.title('3D UMAP Visualization\n(Weighted by Component Variance)')
ax.legend()

# Adjust the viewing angle for better visualization
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig('umap_3d_no_noise.png', bbox_inches='tight', dpi=300)
plt.close()

# Alternative 3D UMAP visualizations without Noise
# 1. Using raw PCA components
raw_data_no_noise = pca_df_zscore_no_outliers[pca_df_zscore_no_outliers['Class'] != 'Noise'].copy()
reducer_3d_raw = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
raw_umap_3d = reducer_3d_raw.fit_transform(raw_data_no_noise[pca_components])

# 2. Using class separation-based weighting
sep_weighted_data = pca_df_zscore_no_outliers.copy()
for i, component in enumerate(pca_components):
    # Calculate Fisher's discriminant ratio for each component
    class_means = []
    class_vars = []
    for class_name in main_class_names:
        class_data = sep_weighted_data[sep_weighted_data['Class'] == class_name][component]
        class_means.append(class_data.mean())
        class_vars.append(class_data.var())
    
    # Calculate between-class variance and within-class variance
    overall_mean = np.mean(class_means)
    between_class_var = np.sum([(m - overall_mean)**2 for m in class_means])
    within_class_var = np.mean(class_vars)
    
    # Fisher's ratio as weight
    fisher_ratio = between_class_var / (within_class_var + 1e-10)  # Add small epsilon to avoid division by zero
    sep_weighted_data[component] = sep_weighted_data[component] * (1.0 + fisher_ratio)

# Filter out Noise class after weighting
sep_weighted_data_no_noise = sep_weighted_data[sep_weighted_data['Class'] != 'Noise'].copy()
reducer_3d_sep = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
sep_umap_3d = reducer_3d_sep.fit_transform(sep_weighted_data_no_noise[pca_components])

# Create figure with two 3D plots side by side
fig = plt.figure(figsize=(20, 8))

# Plot 1: Raw PCA components
ax1 = fig.add_subplot(121, projection='3d')
for class_name in main_class_names:
    mask = raw_data_no_noise['Class'] == class_name
    ax1.scatter(raw_umap_3d[mask, 0], 
               raw_umap_3d[mask, 1], 
               raw_umap_3d[mask, 2],
               c=[class_color_dict[class_name]], 
               label=class_name, 
               alpha=0.6)

ax1.set_xlabel('UMAP1')
ax1.set_ylabel('UMAP2')
ax1.set_zlabel('UMAP3')
ax1.set_title('3D UMAP: Raw PCA Components')
ax1.legend()
ax1.view_init(elev=20, azim=45)

# Plot 2: Class separation weighted
ax2 = fig.add_subplot(122, projection='3d')
for class_name in main_class_names:
    mask = sep_weighted_data_no_noise['Class'] == class_name
    ax2.scatter(sep_umap_3d[mask, 0], 
               sep_umap_3d[mask, 1], 
               sep_umap_3d[mask, 2],
               c=[class_color_dict[class_name]], 
               label=class_name, 
               alpha=0.6)

ax2.set_xlabel('UMAP1')
ax2.set_ylabel('UMAP2')
ax2.set_zlabel('UMAP3')
ax2.set_title('3D UMAP: Class Separation Weighted')
ax2.legend()
ax2.view_init(elev=20, azim=45)

plt.suptitle('Alternative 3D UMAP Visualizations\n(Without Noise Class)', fontsize=14, y=1.05)
plt.tight_layout()
plt.savefig('umap_3d_alternative_approaches.png', bbox_inches='tight', dpi=300)
plt.close()

# BP-focused 3D UMAP visualization
bp_focused_data = pca_df_zscore_no_outliers.copy()

# Calculate weights focusing on BP separation
for i, component in enumerate(pca_components):
    # Get data for each class
    bp_data = bp_focused_data[bp_focused_data['Class'] == 'BP'][component]
    other_data = bp_focused_data[bp_focused_data['Class'].isin(['Whistle', 'Click'])][component]
    
    # Calculate mean and variance for BP and other classes
    bp_mean = bp_data.mean()
    other_mean = other_data.mean()
    bp_var = bp_data.var()
    other_var = other_data.var()
    
    # Calculate separation metrics
    mean_diff = abs(bp_mean - other_mean)
    var_ratio = max(bp_var, other_var) / (min(bp_var, other_var) + 1e-10)
    
    # Calculate weight based on both mean difference and variance ratio
    # Higher weight if BP is well-separated from other classes
    weight = (1.0 + mean_diff) * (1.0 + 1.0/var_ratio)
    
    # Apply weight to the component
    bp_focused_data[component] = bp_focused_data[component] * weight

# Filter out Noise class after weighting
bp_focused_data_no_noise = bp_focused_data[bp_focused_data['Class'] != 'Noise'].copy()

# Try different UMAP parameters for BP-focused visualization
# 1. Standard parameters
reducer_3d_bp1 = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
bp_umap_3d_1 = reducer_3d_bp1.fit_transform(bp_focused_data_no_noise[pca_components])

# 2. More emphasis on local structure
reducer_3d_bp2 = umap.UMAP(n_components=3, random_state=42, n_neighbors=5, min_dist=0.3)
bp_umap_3d_2 = reducer_3d_bp2.fit_transform(bp_focused_data_no_noise[pca_components])

# Create figure with two 3D plots side by side
fig = plt.figure(figsize=(20, 8))

# Plot 1: Standard parameters
ax1 = fig.add_subplot(121, projection='3d')
for class_name in main_class_names:
    mask = bp_focused_data_no_noise['Class'] == class_name
    ax1.scatter(bp_umap_3d_1[mask, 0], 
               bp_umap_3d_1[mask, 1], 
               bp_umap_3d_1[mask, 2],
               c=[class_color_dict[class_name]], 
               label=class_name, 
               alpha=0.6)

ax1.set_xlabel('UMAP1')
ax1.set_ylabel('UMAP2')
ax1.set_zlabel('UMAP3')
ax1.set_title('BP-Focused UMAP\n(n_neighbors=15, min_dist=0.1)')
ax1.legend()
ax1.view_init(elev=20, azim=45)

# Plot 2: Local structure emphasis
ax2 = fig.add_subplot(122, projection='3d')
for class_name in main_class_names:
    mask = bp_focused_data_no_noise['Class'] == class_name
    ax2.scatter(bp_umap_3d_2[mask, 0], 
               bp_umap_3d_2[mask, 1], 
               bp_umap_3d_2[mask, 2],
               c=[class_color_dict[class_name]], 
               label=class_name, 
               alpha=0.6)

ax2.set_xlabel('UMAP1')
ax2.set_ylabel('UMAP2')
ax2.set_zlabel('UMAP3')
ax2.set_title('BP-Focused UMAP\n(n_neighbors=5, min_dist=0.3)')
ax2.legend()
ax2.view_init(elev=20, azim=45)

plt.suptitle('BP-Focused 3D UMAP Visualizations\n(Weighted to Emphasize BP Separation)', fontsize=14, y=1.05)
plt.tight_layout()
plt.savefig('umap_3d_bp_focused.png', bbox_inches='tight', dpi=300)
plt.close()

# Add model evaluation visualizations
def plot_learning_curves(history, save_path='learning_curves.png'):
    """
    Plot training and validation metrics over epochs.
    
    Args:
        history: Keras history object containing training metrics
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_class_metrics(y_true, y_pred, class_names, save_path='class_metrics.png'):
    """
    Plot ROC curves and Precision-Recall curves for each class.
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted probabilities
        class_names: List of class names
        save_path: Path to save the plot
    """
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot ROC curves
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves by Class')
    ax1.legend(loc="lower right")
    ax1.grid(True)
    
    # Plot Precision-Recall curves
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        avg_precision = average_precision_score(y_true[:, i], y_pred[:, i])
        ax2.plot(recall, precision, label=f'{class_name} (AP = {avg_precision:.2f})')
    
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves by Class')
    ax2.legend(loc="lower left")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_confidence_distribution(y_true, y_pred, class_names, save_path='confidence_distribution.png'):
    """
    Plot confidence distribution for correct and incorrect predictions.
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted probabilities
        class_names: List of class names
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, class_name in enumerate(class_names):
        if i < 4:  # Plot first 4 classes
            # Get predictions for this class
            true_positives = y_pred[(y_true[:, i] == 1) & (y_pred[:, i] > 0.5), i]
            false_positives = y_pred[(y_true[:, i] == 0) & (y_pred[:, i] > 0.5), i]
            false_negatives = y_pred[(y_true[:, i] == 1) & (y_pred[:, i] <= 0.5), i]
            
            # Plot histograms
            axes[i].hist(true_positives, bins=20, alpha=0.5, label='True Positives', color='green')
            axes[i].hist(false_positives, bins=20, alpha=0.5, label='False Positives', color='red')
            axes[i].hist(false_negatives, bins=20, alpha=0.5, label='False Negatives', color='blue')
            
            axes[i].set_title(f'Confidence Distribution - {class_name}')
            axes[i].set_xlabel('Prediction Confidence')
            axes[i].set_ylabel('Count')
            axes[i].legend()
            axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_feature_importance(model, input_data, class_names, save_path='feature_importance.png'):
    """
    Plot feature importance using permutation importance.
    
    Args:
        model: Trained model
        input_data: Input data used for prediction
        class_names: List of class names
        save_path: Path to save the plot
    """
    from sklearn.inspection import permutation_importance
    
    # Calculate permutation importance
    result = permutation_importance(model, input_data, y_true, n_repeats=10, random_state=42)
    
    # Plot importance scores
    plt.figure(figsize=(12, 6))
    importance_df = pd.DataFrame(
        result.importances.T,
        columns=[f'Feature {i+1}' for i in range(input_data.shape[1])]
    )
    
    sns.boxplot(data=importance_df)
    plt.title('Feature Importance Scores')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_decision_boundaries(model, X, y, class_names, save_path='decision_boundaries.png'):
    """
    Plot decision boundaries for the first two features.
    
    Args:
        model: Trained model
        X: Input features
        y: True labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    
    # Get predictions for mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundaries
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title('Decision Boundaries')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

# Run all model evaluation visualizations
print("\nGenerating model evaluation visualizations...")

# Assuming we have the model history and predictions
try:
    # Load model history if available
    import json
    with open('model_history.json', 'r') as f:
        history_dict = json.load(f)
        history = type('History', (), {'history': history_dict})()
    
    # Generate learning curves
    print("Generating learning curves...")
    plot_learning_curves(history, save_path='model_learning_curves.png')
    
    # Load model predictions if available
    import numpy as np
    y_true = np.load('y_true.npy')
    y_pred = np.load('y_pred.npy')
    
    # Generate class metrics
    print("Generating class metrics...")
    plot_class_metrics(y_true, y_pred, class_names, save_path='model_class_metrics.png')
    
    # Generate confidence distribution
    print("Generating confidence distribution...")
    plot_confidence_distribution(y_true, y_pred, class_names, save_path='model_confidence_distribution.png')
    
    # Load model and input data if available
    from tensorflow.keras.models import load_model
    model = load_model('multilabel_model.h5')
    X_test = np.load('X_test.npy')
    
    # Generate feature importance
    print("Generating feature importance...")
    plot_feature_importance(model, X_test, class_names, save_path='model_feature_importance.png')
    
    # Generate decision boundaries
    print("Generating decision boundaries...")
    plot_decision_boundaries(model, X_test, y_true, class_names, save_path='model_decision_boundaries.png')
    
    print("\nAll visualizations have been generated and saved!")
    
except FileNotFoundError as e:
    print(f"\nError: Could not find required files. Please ensure the following files exist:")
    print("- model_history.json (model training history)")
    print("- y_true.npy (true labels)")
    print("- y_pred.npy (model predictions)")
    print("- multilabel_model.h5 (trained model)")
    print("- X_test.npy (test data)")
    print(f"\nSpecific error: {str(e)}")
except Exception as e:
    print(f"\nError generating visualizations: {str(e)}")