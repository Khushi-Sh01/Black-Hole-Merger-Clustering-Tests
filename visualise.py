import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
os.makedirs('plots', exist_ok=True)
TRUE_COLORS = {
    'field_binary'    : '#2196F3',  
    'globular_cluster': '#FF9800',  
    'pop3_remnant'    : '#9C27B0', 
}
CLUSTER_COLORS = ['#E53935', '#43A047', '#FB8C00']
pca_df    = pd.read_csv('results/pca_output.csv')
kmeans_df = pd.read_csv('results/kmeans_results.csv')
dbscan_df = pd.read_csv('results/dbscan_results.csv')
raw_data  = pd.read_csv('data/bh_mergers.csv')

fig, ax = plt.subplots(figsize=(8, 6))
for channel, color in TRUE_COLORS.items():
    mask = pca_df['true_channel'] == channel
    ax.scatter(
        pca_df.loc[mask, 'PC1'],
        pca_df.loc[mask, 'PC2'],
        c=color, label=channel.replace('_', ' ').title(),
        alpha=0.5, s=15, edgecolors='none'
    )
ax.set_xlabel('Principal Component 1 (PC1)', fontsize=12)
ax.set_ylabel('Principal Component 2 (PC2)', fontsize=12)
ax.set_title('True BH Merger Channels in PCA Space\n(Answer Key)', fontsize=13)
ax.legend(title='Merger Channel', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/01_true_channels.png', dpi=150)
plt.close()
print("Saved: plots/01_true_channels.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for label in sorted(kmeans_df['kmeans_label'].unique()):
    mask = kmeans_df['kmeans_label'] == label
    axes[0].scatter(
        kmeans_df.loc[mask, 'PC1'],
        kmeans_df.loc[mask, 'PC2'],
        c=CLUSTER_COLORS[label % 3],
        label=f'ML Cluster {label}',
        alpha=0.5, s=15
    )
axes[0].set_title('K-Means: What ML Found', fontsize=12)
axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2')
axes[0].legend(); axes[0].grid(True, alpha=0.3)
for channel, color in TRUE_COLORS.items():
    mask = kmeans_df['true_channel'] == channel
    axes[1].scatter(
        kmeans_df.loc[mask, 'PC1'],
        kmeans_df.loc[mask, 'PC2'],
        c=color, label=channel.replace('_', ' ').title(),
        alpha=0.5, s=15
    )
axes[1].set_title('True Channels (Answer Key)', fontsize=12)
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.suptitle('K-Means Clustering vs. True Channels', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('plots/02_kmeans_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: plots/02_kmeans_comparison.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
unique_labels = sorted(dbscan_df['dbscan_label'].unique())
color_map = {-1: 'gray'}  # -1 = noise in DBSCAN
for i, lbl in enumerate([l for l in unique_labels if l != -1]):
    color_map[lbl] = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
for label in unique_labels:
    mask = dbscan_df['dbscan_label'] == label
    name = 'Noise' if label == -1 else f'Cluster {label}'
    axes[0].scatter(
        dbscan_df.loc[mask, 'PC1'],
        dbscan_df.loc[mask, 'PC2'],
        c=color_map[label],
        label=name, alpha=0.5, s=15
    )
axes[0].set_title('DBSCAN: What ML Found', fontsize=12)
axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

for channel, color in TRUE_COLORS.items():
    mask = dbscan_df['true_channel'] == channel
    axes[1].scatter(
        dbscan_df.loc[mask, 'PC1'],
        dbscan_df.loc[mask, 'PC2'],
        c=color, label=channel.replace('_', ' ').title(),
        alpha=0.5, s=15
    )
axes[1].set_title('True Channels (Answer Key)', fontsize=12)
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.suptitle('DBSCAN Clustering vs. True Channels', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('plots/03_dbscan_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: plots/03_dbscan_comparison.png")

params = ['m1', 'spin1', 'metallicity', 'redshift']
param_labels = ['Primary Mass (M☉)', 'Spin Parameter', 'Metallicity', 'Redshift z']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, (param, label) in enumerate(zip(params, param_labels)):
    for channel, color in TRUE_COLORS.items():
        subset = raw_data[raw_data['channel'] == channel][param]
        axes[i].hist(
            subset, bins=30, alpha=0.5,
            color=color, label=channel.replace('_', ' ').title(),
            density=True  # Normalize so all three fit on same scale
        )
    axes[i].set_xlabel(label, fontsize=11)
    axes[i].set_ylabel('Normalized Count', fontsize=11)
    axes[i].set_title(f'Distribution of {label}', fontsize=11)
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Physical Parameter Distributions per Merger Channel', fontsize=13)
plt.tight_layout()
plt.savefig('plots/04_parameter_distributions.png', dpi=150)
plt.close()
print("   01_true_channels.png        → The answer key")
print("   02_kmeans_comparison.png    → K-Means vs truth")
print("   03_dbscan_comparison.png    → DBSCAN vs truth")
print("   04_parameter_distributions.png → Why channels are separable")
