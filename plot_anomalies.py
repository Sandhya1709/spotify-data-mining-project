
import matplotlib.pyplot as plt

# Isolation Forest results
labels = ["Normal Tracks", "Anomalies"]
counts = [7942, 2058]
percentages = [79.42, 20.58]

# Create bar chart
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, counts)

# Add title and labels
plt.title("Anomaly Detection Using Isolation Forest")
plt.xlabel("Track Category")
plt.ylabel("Number of Tracks")

# Display count and percentage above each bar
for bar, count, percentage in zip(bars, counts, percentages):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 100,
        f"{count:,}\n({percentage}%)",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

# Save chart for PowerPoint
plt.savefig("isolation_forest_chart.png", dpi=300, bbox_inches="tight")

# Display chart
plt.show()
