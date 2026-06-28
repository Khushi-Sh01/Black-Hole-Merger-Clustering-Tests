import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, confusion_matrix, silhouette_score)       
import joblib
import os
raw_data  = pd.read_csv('data/bh_mergers.csv')
kmeans_df = pd.read_csv('results/kmeans_results.csv')
dbscan_df = pd.read_csv('results/dbscan_results.csv')
scaler    = joblib.load('models/scaler.pkl')
features  = ['m1', 'm2', 'spin1', 'spin2', 'metallicity',
             'redshift', 'mass_ratio', 'total_mass']
X_scaled  = scaler.transform(raw_data[features].values)
true_labels = raw_data['channel']
print(f"Total mergers simulated : {len(raw_data)}")
print(f"Merger channels         : 3")
print(f"Features used           : {len(features)}")
print(f"\nChannel breakdown:")
for ch, count in raw_data['channel'].value_counts().items():
    print(f"  {ch:<20}: {count} mergers")
summary = raw_data.groupby('channel')[['m1','spin1','metallicity','redshift']].mean()
summary.columns = ['Mean m1 (M☉)', 'Mean Spin', 'Mean Metallicity', 'Mean Redshift']
print(summary.round(4).to_string())
print("\nK-MEANS RESULTS")
km_labels = kmeans_df['kmeans_label']
ari_km    = adjusted_rand_score(true_labels, km_labels)
sil_km    = silhouette_score(X_scaled, km_labels)
print(f"Adjusted Rand Index (ARI) : {ari_km:.4f}")
print(f"Silhouette Score          : {sil_km:.4f}")
print(f"\nConfusion Matrix (rows=true channel, cols=ML cluster):")
cm_km = pd.crosstab(
    kmeans_df['true_channel'],
    kmeans_df['kmeans_label'],
    rownames=['True Channel'],
    colnames=['ML Cluster']
)
print(cm_km.to_string())
print("\n Each row should be dominated by ONE column.")
print("  If it is, the ML recovered that channel successfully.")
print("\n DBSCAN RESULTS")
print("-"*40)
db_labels = dbscan_df['dbscan_label']
n_clusters = len(set(db_labels)) - (1 if -1 in db_labels.values else 0)
n_noise    = (db_labels == -1).sum()
ari_db     = adjusted_rand_score(true_labels, db_labels)
print(f"Clusters found            : {n_clusters}")
print(f"Noise points              : {n_noise} ({100*n_noise/len(db_labels):.1f}%)")
print(f"Adjusted Rand Index (ARI) : {ari_db:.4f}")

# Only compute silhouette if we have valid clusters
valid_mask = db_labels != -1
if valid_mask.sum() > 1 and len(set(db_labels[valid_mask])) > 1:
    sil_db = silhouette_score(X_scaled[valid_mask], db_labels[valid_mask])
    print(f"Silhouette Score          : {sil_db:.4f} (excluding noise)")
print(f"\nConfusion Matrix:")
cm_db = pd.crosstab(
    dbscan_df['true_channel'],
    dbscan_df['dbscan_label'],
    rownames=['True Channel'],
    colnames=['DBSCAN Cluster']
)
print(cm_db.to_string())

print(f"{'Metric':<30} {'K-Means':>10} {'DBSCAN':>10}")
print("-"*50)
print(f"{'Adjusted Rand Index':<30} {ari_km:>10.4f} {ari_db:>10.4f}")
print(f"{'Clusters Found':<30} {'3':>10} {n_clusters:>10}")
print(f"{'Noise Points':<30} {'0':>10} {n_noise:>10}")
