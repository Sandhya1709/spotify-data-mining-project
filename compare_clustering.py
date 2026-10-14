
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# Load the dataset
df = pd.read_csv("data.csv")

features = [
    "valence", "acousticness", "danceability",
    "energy", "instrumentalness", "liveness",
    "loudness", "speechiness", "tempo"
]

# Use the same 10,000 tracks for both algorithms
sample = df.sample(n=10000, random_state=42)
X = StandardScaler().fit_transform(sample[features])

# Run K-Means
start = time.time()

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)
kmeans_labels = kmeans.fit_predict(X)

kmeans_time = time.time() - start
kmeans_score = silhouette_score(X, kmeans_labels)

# Run DBSCAN
start = time.time()

dbscan = DBSCAN(eps=0.8, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

dbscan_time = time.time() - start

# Count DBSCAN clusters and noise points
noise_mask = dbscan_labels == -1
cluster_count = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
noise_count = noise_mask.sum()
noise_percentage = noise_count / len(X) * 100

# Calculate DBSCAN silhouette score, excluding noise
normal_mask = ~noise_mask
normal_labels = dbscan_labels[normal_mask]
normal_X = X[normal_mask]

if len(set(normal_labels)) > 1:
    dbscan_score = silhouette_score(normal_X, normal_labels)
else:
    dbscan_score = None

# Print comparison
print("\n--- Clustering Comparison ---")
print(f"K-Means clusters: 4")
print(f"K-Means silhouette score: {kmeans_score:.4f}")
print(f"K-Means execution time: {kmeans_time:.2f} seconds")

print(f"\nDBSCAN clusters: {cluster_count}")
print(f"DBSCAN noise points: {noise_count}")
print(f"DBSCAN noise percentage: {noise_percentage:.2f}%")
print(f"DBSCAN silhouette score (excluding noise): {dbscan_score}")
print(f"DBSCAN execution time: {dbscan_time:.2f} seconds")
