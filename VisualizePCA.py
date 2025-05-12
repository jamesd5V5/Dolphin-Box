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