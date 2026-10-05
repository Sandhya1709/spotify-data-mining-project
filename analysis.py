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

'''import matplotlib.pyplot as plt

# Plot distributions of selected audio features
df[selected_features].hist(figsize=(12, 10), bins=30)

plt.suptitle("Distribution of Selected Audio Features")
plt.tight_layout()
plt.show()'''

# Select features for clustering
X = df[selected_features]

print("\nClustering data shape:")
print(X.shape)

from sklearn.preprocessing import StandardScaler

# Standardize the clustering features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled data shape:")
print(X_scaled.shape)

print("\nChecking for NaN values in clustering data:")
print(X.isnull().sum().sum())

print("\nChecking for infinite values:")
print(X.replace([float("inf"), float("-inf")], pd.NA).isnull().sum().sum())