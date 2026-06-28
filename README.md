# Black Hole Merger Population Clustering using Machine Learning

Gravitational wave observatories such as **LIGO**, **Virgo**, and **KAGRA** have detected hundreds of black hole mergers. Each detected event contains physical information such as the masses of the black holes, the metallicity of their birth environment, and the cosmological redshift at which the merger occurred. One of the major challenges in modern astrophysics is determining **how these black holes formed**. Different formation mechanisms produce black holes with different physical properties. These formation mechanisms are commonly called **formation channels**. The goal of this project is to simply see whether **unsupervised machine learning** can automatically discover these formation channels using only observable physical parameters, **without being given any labels during training**. Instead of telling the algorithm that, this merger came from a globular cluster, we simply provide the physical features: Black hole mass, Metallicity and Redshift and ask:

> **Can machine learning discover the hidden populations on its own?**

# A Little Background

Black holes can form through several different evolutionary pathways. For this project we simulate three well-known formation channels.

| Formation Channel                 | Mass Range | Metallicity       | Typical Redshift |
| --------------------------------- | ---------- | ----------------- | ---------------- |
| Field Binaries                    | 5–40 M☉    | High (solar-like) | z = 0–3          |
| Globular Clusters                 | 20–80 M☉   | Low               | z = 0–6          |
| Population III (Pop III) Remnants | 50–150 M☉  | Nearly zero       | z = 6–20         |

1. Field Binary Evolution: Two massive stars are born together as a binary system. Both stars evolve, explode as supernovae, and eventually become black holes while remaining gravitationally bound. Over billions of years they lose orbital energy through gravitational-wave emission until they merge. Some identifiable Characteristics: relatively low masses, metal-rich environment,mostly found at lower redshift,common in nearby galaxies

2. Globular Cluster Formation:Globular clusters are dense collections containing hundreds of thousands of stars.Inside these environments, black holes frequently interact gravitationally.Repeated encounters can pair unrelated black holes into binaries that eventually merge. Some identifiable Characteristics: intermediate masses,lower metallicity,larger spread in redshift.

3. Population III Remnants: Population III stars were the **first generation of stars** in the Universe, as believed in theory.Because the early Universe contained almost no heavy elements, these stars could become extremely massive. After their deaths they leave behind massive black holes. Characteristics: highest masses, almost zero metallicity,high redshift (early Universe), possible seeds of supermassive black holes

# The Idea!

In real gravitational-wave catalogs, we do **not** know the true formation channel of each merger.So, this is naturally an **unsupervised learning** problem. The objective is to determine whether the merger population naturally separates into distinct clusters based only on their physical properties. Since the true formation history of real mergers is unknown, we just create a synthetic dataset.The data are generated from distributions representing the three formation channels above. Ground-truth labels are stored only for evaluation purposes and are never used during model training.

---

# Machine Learning Pipeline

We follow a simple workflow from data generation to final analysis. First, a synthetic dataset of black hole mergers is created, where each merger has physical properties such as mass, metallicity, and redshift. Since these features have different ranges of values, they are scaled using **StandardScaler** so that each feature contributes equally during clustering. Next, **Principal Component Analysis (PCA)** is used to reduce the data to two dimensions, making it easier to visualize the merger population. The clustering algorithms **K-Means** and **DBSCAN** are then applied to the scaled data to group similar mergers together without using the true formation channel labels. At last, the quality of the clusters is measured using the **Adjusted Rand Index (ARI)** and **Silhouette Score** and different plots are generated to help understand how well the machine learning algorithms separate the different black hole merger populations.
---

1. StandardScaler: The three features have very different numerical ranges. Example:Mass:5 – 150|Metallicity:0 – 1|Redshift:
0 – 20. Algorithms like K-Means compute Euclidean distances. Without scaling, the mass feature would dominate every distance calculation simply because its numerical values are much larger.StandardScaler transforms every feature to 
* mean = 0
* standard deviation = 1

using x'=x-mu\sigma so every feature contributes equally during clustering.

2. Principal Component Analysis (PCA):Our dataset has three dimensions.PCA projects the data into two principal components while preserving as much variance as possible.The first principal component captures the largest variation in the data.The second captures the next largest variation while remaining orthogonal to the first.This allows us to create informative 2D scatter plots of the merger population.PCA is used only for visualization.The clustering algorithms are trained using the original scaled features.

3. K-Means Clustering:K-Means is a centroid-based clustering algorithm.The algorithm begins by randomly placing **K cluster centers**.It then repeatedly performs two steps. Assign every merger to its nearest centroid then move each centroid to the average position of all assigned points.These two steps repeat until the centroids stop changing significantly.

4. DBSCAN:DBSCAN stands for **Density-Based Spatial Clustering of Applications with Noise.**Instead of assuming spherical clusters, DBSCAN groups points according to local density.It requires two parameters:**eps** which is the maximum neighborhood radius and **min_samples**, the minimum number of nearby points needed to form a dense region.
---

# Evaluation Metrics

A. Adjusted Rand Index (ARI):ARI measures how closely predicted clusters match the true formation channels.Unlike the ordinary Rand Index, ARI corrects for agreement occurring purely by chance. Interpretation:ARI = 1.0: Perfect clustering|ARI = 0:Random clustering|ARI < 0:Worse than random

B. Silhouette Score: The Silhouette Score evaluates clustering quality **without using labels**.Each point receives a score between -1 and +1. Interpretation:+1: Well-separated clusters|0:Overlapping clusters|Negative:Likely assigned to the wrong cluster.


---

# Installation

Install the required Python packages.

```bash
pip install numpy pandas scikit-learn matplotlib joblib
```

---

# Running the Project

Generate the synthetic merger population.

```bash
python synthesizing_data.py
```

Train the clustering algorithms.

```bash
python model_training.py
```

Evaluate clustering performance.

```bash
python results.py
```

Generate all visualizations.

```bash
python visualize.py
```

---


# References

1. Volonteri, M. (2008), *Formation of Supermassive Black Holes*, MNRAS.
2. Abbott et al., LIGO-Virgo-KAGRA Collaboration — Gravitational Wave Catalogs (GWTC).
3. Scikit-learn Documentation — K-Means, DBSCAN, PCA, StandardScaler.
