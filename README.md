# Acoustic Profiling and Anomaly Detection in Large-Scale Music Catalogs

## Project Overview
This project explores audio characteristics in Spotify tracks to discover groups of similar tracks and identify unusual tracks without relying only on predefined music genres.

## Research Question
To what extent can audio characteristics reveal meaningful relationships between songs and help identify unusual tracks in a large-scale music dataset?

## Dataset
- **Source:** [Spotify Dataset (1921–2020, 160k+ Tracks)](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-1921-2020-160k-tracks)
- **Number of tracks:** 170,653
- **Audio features:** Valence, acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, and tempo.
- Dataset files are not included in this repository.

## Methods
- **Exploratory Data Analysis (EDA):** Examined feature distributions and potential outliers.
- **StandardScaler:** Standardized numerical audio features.
- **K-Means:** Grouped tracks into four clusters.
- **DBSCAN:** Identified density-based clusters and noise points in a 10,000-track sample.
- **Isolation Forest:** Detected potentially unusual tracks in the same sample.

## Main Findings
- K-Means identified four groups with different audio characteristics.
- DBSCAN identified 21 clusters and classified 38.03% of the sample as noise.
- Isolation Forest classified 2,058 tracks (20.58%) as anomalies.
- Anomalies represent unusual feature combinations and are not necessarily errors.

## Project Files
- `analysis.py` — Main analysis and clustering workflow.
- `compare_clustering.py` — Clustering comparison.
- `compare_anomalies.py` — Anomaly comparison.
- `inspect_common_anomalies.py` — Inspects common anomalies.
- `top_anomalies.py` — Examines top anomalies.
- `plot_anomalies.py` — Generates the anomaly detection chart.
- `isolation_forest_chart.png` — Anomaly detection chart.
- `isolation_forest_results.csv` — Isolation Forest results.
- `common_anomalies.csv` — Common anomaly results.

## Requirements
Python 3 and the required libraries used in the scripts, including pandas, NumPy, and Matplotlib. Install any additional libraries imported by the scripts before running them.

## How to Run
1. Clone or download this repository.
2. Install the required Python libraries.
3. Download the dataset from Kaggle and place the required CSV files in the project folder.
4. Run the relevant Python scripts from the terminal.

## Note
This project was developed for the Data Mining I course. The clustering and anomaly detection results depend on the selected algorithms, parameters, and sample size.