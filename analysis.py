import pandas as pd

# Load the dataset
df = pd.read_csv("data.csv")

# Display the first 5 rows
print(df.head())

# Dataset shape
print("\nDataset shape:")
print(df.shape)

# Column names
print("\nColumn names:")
print(df.columns.tolist())

# Data types
print("\nData types:")
print(df.dtypes)

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Selected audio features
print("\nSelected audio features:")
print(df[
    [
        "valence",
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "tempo"
    ]
].describe())

# Check for potential outliers using IQR
print("\nPotential outliers:")

selected_features = [
    "valence",
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo"
]

for column in selected_features:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    print(column, ":", len(outliers))

import matplotlib.pyplot as plt

# Plot distributions of selected audio features
df[selected_features].hist(figsize=(12, 10), bins=30)

plt.suptitle("Distribution of Selected Audio Features")
plt.tight_layout()
plt.show()



# Boxplots to inspect potential outliers
plt.figure(figsize=(14, 8))

df[selected_features].boxplot()
plt.title("Boxplots of Selected Audio Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Select features for clustering
X = df[selected_features]

print("\nClustering data shape:")
print(X.shape)

from sklearn.preprocessing import StandardScaler

# Standardize the clustering features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)




# Boxplots of standardized audio features
X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=selected_features
)

plt.figure(figsize=(14, 8))
X_scaled_df.boxplot()
plt.title("Boxplots of Standardized Audio Features")
plt.xticks(rotation=45)
plt.ylabel("Standardized value (z-score)")
plt.tight_layout()
plt.show()




print("\nScaled data shape:")
print(X_scaled.shape)

print("\nChecking for NaN values in clustering data:")
print(X.isnull().sum().sum())

print("\nChecking for infinite values:")
print(X.replace([float("inf"), float("-inf")], pd.NA).isnull().sum().sum())





import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("data.csv")

# Select audio features
selected_features = [
    "valence",
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo"
]

# Prepare and scale features
X = df[selected_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method
inertia_values = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia_values, marker="o")
plt.title("Elbow Method for K-Means")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()
plt.show()

print("Elbow method completed.")




# Final K-Means model
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

# Assign each track to a cluster
df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

# Display cluster sizes
print("\nK-Means Cluster Sizes:")
print(df["kmeans_cluster"].value_counts().sort_index())

print("\nK-Means clustering completed.")




# Analyze average audio features in each cluster
cluster_profiles = df.groupby("kmeans_cluster")[
    selected_features
].mean()

print("\nAverage Audio Features per Cluster:")
print(cluster_profiles.round(3))

# Visualize cluster profiles
cluster_profiles.T.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Average Audio Features by K-Means Cluster")
plt.xlabel("Audio Features")
plt.ylabel("Average Feature Value")
plt.xticks(rotation=45)
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()




import seaborn as sns

# Create a DataFrame of standardized features
X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=selected_features,
    index=df.index
)

# Calculate average standardized feature values per cluster
scaled_cluster_profiles = X_scaled_df.groupby(
    df["kmeans_cluster"]
).mean()

# Plot heatmap
plt.figure(figsize=(12, 5))

sns.heatmap(
    scaled_cluster_profiles,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Standardized Audio Feature Profiles by K-Means Cluster")
plt.xlabel("Audio Features")
plt.ylabel("K-Means Cluster")
plt.tight_layout()
plt.show()





from sklearn.cluster import DBSCAN

# Use a reproducible sample of 10,000 tracks
sample_size = 10000
sample_indices = df.sample(
    n=sample_size, random_state=42
).index

X_sample = X_scaled[df.index.get_indexer(sample_indices)]

print("Running DBSCAN on 10,000 tracks...", flush=True)


dbscan = DBSCAN(eps=0.8, min_samples=5)

sample_labels = dbscan.fit_predict(X_sample)


import pandas as pd

cluster_sizes = pd.Series(sample_labels).value_counts()

print("\n10 Largest DBSCAN Clusters:")
print(cluster_sizes[cluster_sizes.index != -1].head(10))



n_clusters = len(set(sample_labels)) - (
    1 if -1 in sample_labels else 0
)
n_noise = (sample_labels == -1).sum()

print("\nDBSCAN Results (10,000-track sample):")
print("Number of clusters:", n_clusters)
print("Number of noise points:", n_noise)
print(f"Noise percentage: {n_noise / sample_size * 100:.2f}%")




from sklearn.ensemble import IsolationForest

# Use the same 10,000-track sample
print("\nRunning Isolation Forest...", flush=True)

isolation_forest = IsolationForest(
    contamination="auto",
    random_state=42
)

anomaly_labels = isolation_forest.fit_predict(X_sample)

# -1 means anomaly; 1 means normal
df_sample = df.loc[sample_indices].copy()
df_sample["anomaly_label"] = anomaly_labels

# Count normal tracks and anomalies
print("\nIsolation Forest Results:")
print(df_sample["anomaly_label"].value_counts())

n_anomalies = (anomaly_labels == -1).sum()
print("Number of anomalies:", n_anomalies)
print(f"Anomaly percentage: {n_anomalies / len(X_sample) * 100:.2f}%")




# Save Isolation Forest results for future use
df_sample["anomaly_score"] = isolation_forest.decision_function(X_sample)

df_sample.to_csv("isolation_forest_results.csv", index=False)

print("Isolation Forest results saved successfully!")



import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reduce the 9 audio features to 2 dimensions for visualisation
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_sample)


# Plot DBSCAN clusters
plt.figure(figsize=(10, 6))

plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=sample_labels,
    cmap="tab20",
    s=5
)

plt.title("DBSCAN Clusters (10,000-Track Sample)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster Label")
plt.tight_layout()
plt.show()





