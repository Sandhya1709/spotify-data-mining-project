
import pandas as pd

df_sample = pd.read_csv("isolation_forest_results.csv")

top_20 = df_sample.sort_values(
    "anomaly_score",
    ascending=True
).head(20)


columns = [
    col for col in [
        "name", "artists", "year",
        "valence", "acousticness", "danceability",
        "energy", "instrumentalness", "liveness",
        "loudness", "speechiness", "tempo",
        "anomaly_score", "anomaly_label"
    ]
    if col in top_20.columns
]


print(top_20[columns].to_string(index=False))
