
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Load the saved Isolation Forest results
df_sample = pd.read_csv("isolation_forest_results.csv")

# Select the same nine audio features
features = [
    "valence", "acousticness", "danceability",
    "energy", "instrumentalness", "liveness",
    "loudness", "speechiness", "tempo"
]

# Standardize the features
X_sample = StandardScaler().fit_transform(df_sample[features])

# Apply DBSCAN using eps=1.0
dbscan = DBSCAN(eps=1.0, min_samples=5)
df_sample["dbscan_cluster"] = dbscan.fit_predict(X_sample)

# Compare DBSCAN noise and Isolation Forest anomalies
df_sample["dbscan_noise"] = df_sample["dbscan_cluster"] == -1
df_sample["if_anomaly"] = df_sample["anomaly_label"] == -1

print("\nComparison of Anomaly Detection Methods:")
print(pd.crosstab(
    df_sample["dbscan_noise"],
    df_sample["if_anomaly"],
    rownames=["DBSCAN noise"],
    colnames=["Isolation Forest anomaly"]
))

both = (df_sample["dbscan_noise"] & df_sample["if_anomaly"]).sum()
print("\nTracks flagged by both methods:", both)





# Select tracks flagged by both methods
common_anomalies = df_sample[
    (df_sample["dbscan_noise"]) &
    (df_sample["if_anomaly"])
]

# Save them to a CSV file
common_anomalies.to_csv(
    "common_anomalies.csv",
    index=False
)

print("\nCommon anomalies saved to common_anomalies.csv")
print("Number of tracks saved:", len(common_anomalies))
