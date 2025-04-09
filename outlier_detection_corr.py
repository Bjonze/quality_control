import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap.umap_ as umap
from scipy.stats import chi2
from matplotlib.patches import Ellipse

# ------------------------------
# 1. Data Loading and Preprocessing
# ------------------------------

laa_measures_path = "/data/Data/LAAAnalysis-08-04-2025/info/laa_measures-statistics-04-04-2025-19-19-22_final_measurements.csv"
fig_save_dir = "/storage/code/quality_control/figures"
os.makedirs(fig_save_dir, exist_ok=True)
outlier_txt = "/storage/code/quality_control/outlier_filenames.txt"

if laa_measures_path.endswith(".json"):
    with open(laa_measures_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
elif laa_measures_path.endswith(".csv"):
    df = pd.read_csv(laa_measures_path, sep=",", header=0)

df = df.drop(columns=["estimated_bifurcations"], errors="ignore")
filenames = df["scan_name"]
df_features = df.drop(columns=["scan_name", "dims_x", "dims_y", "dims_z", "Unnamed: 46"], errors="ignore")
df_features = df_features.fillna(df_features.median())

scaler = StandardScaler()
X = scaler.fit_transform(df_features)

# ------------------------------
# 2. Dimensionality Reduction for Visualization and Outlier Detection
# ------------------------------

# 2a. For visualization: a 2D PCA projection.
pca_vis = PCA(n_components=2)
X_pca_vis = pca_vis.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca_vis[:, 0], X_pca_vis[:, 1], alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("2D PCA Projection of LAA Shape Measures")
plt.savefig(os.path.join(fig_save_dir, 'PCA_LAA_Shape_Measures.png'))
plt.close()

# 2b. Plot cumulative explained variance for all PCA components.
pca_all = PCA()
pca_all.fit(X)
cum_explained = np.cumsum(pca_all.explained_variance_ratio_) * 100  # in percentage
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(cum_explained)+1), cum_explained, marker='o', linestyle='-')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("Cumulative Explained Variance by PCA Components")
plt.axhline(95, color='red', linestyle='--', label="95% Threshold")
plt.legend()
plt.savefig(os.path.join(fig_save_dir, 'PCA_Cumulative_Explained_Variance.png'))
plt.close()

# 2c. For outlier detection: Retain enough PCA components to explain 95% variance.
pca_full = PCA(n_components=0.95, svd_solver='full')
X_pca_full = pca_full.fit_transform(X)
n_components = X_pca_full.shape[1]
print(f"Number of PCA components retained to cover 95% variance: {n_components}")

# ------------------------------
# 3. Unsupervised Outlier Detection (Correlation-Agnostic Methods)
# ------------------------------

# Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_labels = iso_forest.fit_predict(X)
df["IsolationForest"] = iso_labels

# One-Class SVM
ocsvm = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
ocsvm_labels = ocsvm.fit_predict(X)
df["OneClassSVM"] = ocsvm_labels

# DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)
df["DBSCAN"] = dbscan_labels

# ------------------------------
# 4. Mahalanobis Distance in PCA Space (95% variance)
# ------------------------------

# Compute the mean and covariance matrix in the PCA space that retains 95% variance.
mean_full = np.mean(X_pca_full, axis=0)
cov_full = np.cov(X_pca_full, rowvar=False)
inv_cov_full = np.linalg.inv(cov_full)

def mahalanobis_distance(x, mean, inv_cov):
    diff = x - mean
    return np.sqrt(diff.T @ inv_cov @ diff)

mdist_full = np.array([mahalanobis_distance(x, mean_full, inv_cov_full) for x in X_pca_full])
alpha = 0.05  # significance level (e.g., 1%)
threshold_full = np.sqrt(chi2.ppf(1 - alpha, df=n_components))
print(f"Mahalanobis distance threshold (full PCA space, alpha={alpha}): {threshold_full:.3f}")

flag_mahal_full = (mdist_full > threshold_full).astype(int)
df["Mahalanobis_Outlier"] = flag_mahal_full

# ------------------------------
# 5. Combine Outlier Flags (Using Mahalanobis Instead of Z-score)
# ------------------------------

flag_iso = (df["IsolationForest"] == -1).astype(int)
flag_ocsvm = (df["OneClassSVM"] == -1).astype(int)
flag_dbscan = (df["DBSCAN"] == -1).astype(int)
flag_mahal = df["Mahalanobis_Outlier"]

df["Outlier_Flag"] = (flag_iso + flag_ocsvm + flag_dbscan + flag_mahal) >= 2

print("Total samples:", len(df))
print("Outliers flagged by combined methods:", df["Outlier_Flag"].sum())

# ------------------------------
# 6. Visualization in 2D PCA Space with Mahalanobis Ellipse (for Visualization Only)
# ------------------------------

# For visualization, use the 2D PCA projection.
mean_vis = np.mean(X_pca_vis, axis=0)
cov_vis = np.cov(X_pca_vis, rowvar=False)
inv_cov_vis = np.linalg.inv(cov_vis)

alpha_vis = 0.05
threshold_2d = np.sqrt(chi2.ppf(1 - alpha_vis, df=2))
print(f"2D Mahalanobis threshold (alpha={alpha_vis}, df=2): {threshold_2d:.3f}")

# Create a binary flag for visualization using the 2D PCA space.
flag_mahal_2d = (np.array([mahalanobis_distance(x, mean_vis, inv_cov_vis) for x in X_pca_vis]) > threshold_2d).astype(int)

inlier_mask = df["Outlier_Flag"] == False
outlier_mask = df["Outlier_Flag"] == True

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X_pca_vis[inlier_mask, 0], X_pca_vis[inlier_mask, 1], c="blue", label="Inliers", alpha=0.7)
ax.scatter(X_pca_vis[outlier_mask, 0], X_pca_vis[outlier_mask, 1], c="red", label="Outliers", alpha=0.7)

# Draw the Mahalanobis ellipse based on the 2D PCA covariance.
eigvals, eigvecs = np.linalg.eigh(cov_vis)
angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
width, height = 2 * threshold_2d * np.sqrt(eigvals)
ellipse = Ellipse(xy=mean_vis, width=width, height=height, angle=angle,
                  edgecolor='black', fc='None', lw=2, ls='--',
                  label=f'2D Chi2 boundary (alpha={alpha_vis})')
ax.add_patch(ellipse)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("2D PCA Projection with Mahalanobis Ellipse (Visualization Only)")
ax.legend()
plt.savefig(os.path.join(fig_save_dir, 'PCA_Mahalanobis_Ellipse.png'))
plt.close()

# print("Outlier Filenames (combined methods):")
# print(filenames[outlier_mask].values)
#save the outlier filenames as txt file
outlier_filenames = filenames[outlier_mask].values
with open(outlier_txt, "w") as f:
    for filename in outlier_filenames:
        f.write(filename + "\n")

# ------------------------------
# 7. TSNE and UMAP Projections with Outliers Highlighted
# ------------------------------

# TSNE Projection
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[inlier_mask, 0], X_tsne[inlier_mask, 1], c="blue", label="Inliers", alpha=0.7)
plt.scatter(X_tsne[outlier_mask, 0], X_tsne[outlier_mask, 1], c="red", label="Outliers", alpha=0.7)
plt.xlabel("t-SNE Dim1")
plt.ylabel("t-SNE Dim2")
plt.title("t-SNE Projection of LAA Shape Measures with Outliers")
plt.legend()
plt.savefig(os.path.join(fig_save_dir, 'TSNE_LAA_Shape_Measures_with_Outliers.png'))
plt.close()

# UMAP Projection
umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(X)
plt.figure(figsize=(8, 6))
plt.scatter(X_umap[inlier_mask, 0], X_umap[inlier_mask, 1], c="blue", label="Inliers", alpha=0.7)
plt.scatter(X_umap[outlier_mask, 0], X_umap[outlier_mask, 1], c="red", label="Outliers", alpha=0.7)
plt.xlabel("UMAP Dim1")
plt.ylabel("UMAP Dim2")
plt.title("UMAP Projection of LAA Shape Measures with Outliers")
plt.legend()
plt.savefig(os.path.join(fig_save_dir, 'UMAP_LAA_Shape_Measures_with_Outliers.png'))
plt.close()

# ------------------------------
# 8. Additional Visualization: Feature Distributions with Outliers Marked
# ------------------------------

selected_features = ['volume', 'tortuosity', 'radii_mean', 'normalized_shape_index']
df_features_with_flag = df_features.copy()
df_features_with_flag["Outlier_Flag"] = df["Outlier_Flag"]

for feature in selected_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df_features_with_flag, x="Outlier_Flag", y=feature)
    plt.title(f"{feature} distribution (0: Inlier, 1: Outlier)")
    plt.savefig(os.path.join(fig_save_dir, f"{feature}_distribution.png"))
    plt.close()
