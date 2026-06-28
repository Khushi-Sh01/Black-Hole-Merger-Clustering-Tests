# model_training.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler   
from sklearn.cluster import KMeans, DBSCAN        
from sklearn.metrics import adjusted_rand_score   
from sklearn.decomposition import PCA              
import joblib           #To save outputs                           
import os
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
data = pd.read_csv('data/bh_mergers.csv')
# We drop the 'channel' column (the true label)
features = ['m1', 'm2', 'spin1', 'spin2', 'metallicity',
            'redshift', 'mass_ratio', 'total_mass']
X = data[features].values          
true_labels = data['channel']      
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   
joblib.dump(scaler, 'models/scaler.pkl')
# PCA = Principal Component Analysis for 8D to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
variance_explained = pca.explained_variance_ratio_.sum() * 100
print(f"{variance_explained:.1f}% of variance captured in 2D")
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['true_channel'] = true_labels.values
pca_df.to_csv('results/pca_output.csv', index=False)
joblib.dump(pca, 'models/pca.pkl')

# MODEL 1: K-Means Clustering
kmeans = KMeans( n_clusters=3, random_state=42, n_init=10  )
kmeans_labels = kmeans.fit_predict(X_scaled)
joblib.dump(kmeans, 'models/kmeans.pkl')
ari_kmeans = adjusted_rand_score(true_labels, kmeans_labels)
print(f"K-Means ARI score: {ari_kmeans:.3f}")
kmeans_df = pd.DataFrame({
    'kmeans_label': kmeans_labels,
    'true_channel': true_labels.values,
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1]
})
kmeans_df.to_csv('results/kmeans_results.csv', index=False)
# MODEL 2: DBSCAN Clustering
dbscan = DBSCAN(
    eps=0.8,            # Max distance to be neighbors 
    min_samples=10      # Min 10 points to be a core point
)
dbscan_labels = dbscan.fit_predict(X_scaled)
n_clusters_found = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
print(f"   Clusters found: {n_clusters_found}")
print(f"   Noise points: {n_noise} ({100*n_noise/len(dbscan_labels):.1f}%)")
joblib.dump(dbscan, 'models/dbscan.pkl')
ari_dbscan = adjusted_rand_score(true_labels, dbscan_labels)
print(f"   DBSCAN ARI score: {ari_dbscan:.3f}")
dbscan_df = pd.DataFrame({
    'dbscan_label': dbscan_labels,
    'true_channel': true_labels.values,
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1]
})
dbscan_df.to_csv('results/dbscan_results.csv', index=False)
print(f"Total mergers: {len(data)}")
print(f"Features used: {features}")
print(f"PCA variance captured: {variance_explained:.1f}%")
print(f"K-Means ARI: {ari_kmeans:.3f}")
print(f"DBSCAN ARI:  {ari_dbscan:.3f}")
