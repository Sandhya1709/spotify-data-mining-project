
import pandas as pd

# Load tracks flagged by both methods
df = pd.read_csv("common_anomalies.csv")

# Show the 20 tracks with the lowest Isolation Forest anomaly scores
df = df.sort_values("anomaly_score", ascending=True)

columns_to_show = [
    "name",
    "artists",
    "year",
    "speechiness",
    "instrumentalness",
    "acousticness",
    "energy",
    "loudness",
    "tempo",
    "anomaly_score"
]

print(df[columns_to_show].head(20).to_string(index=False))
