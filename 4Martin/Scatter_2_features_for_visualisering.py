import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
output_folder = BASE_DIR / "4Martin"

# kNN-modell, k = 5
k = 5

# Modellen skal trenes på parameterne spectral_rolloff_mean (index 10), mfcc_1_mean (index 41)
# spectral_centroid_mean (index 6), tempo (index 40) GenreID (index 65)


# Shape: 990


# cols = [
#     "spectral_rolloff_mean",
#     "mfcc_1_mean",
#     "spectral_centroid_mean",
#     "tempo",
#     "Genre"
# ]

cols = [
    "spectral_rolloff_mean",
    "mfcc_1_mean",
    "Genre"
]


data_file = BASE_DIR / "Music files" / "GenreClassData_30s.txt"
df = pd.read_csv(data_file, sep="\t", usecols=cols)
features = df.drop(columns=["Genre"]).values
labels = df["Genre"].values
train_features = features[:794]
print(f"train_features.shape: {train_features.shape}")
train_labels = labels[:794]

genre1 = "blues"
genre2 = "classical"

mask1 = train_labels == genre1
mask2 = train_labels == genre2

test_features_all = features[794:]
test_labels_all = labels[794:]

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features_all = scaler.transform(test_features_all)

mu_blues = np.mean(train_features[mask1], axis=0)
mu_classical = np.mean(train_features[mask2], axis=0)
C_blues = np.cov(train_features[mask1], rowvar=False)
C_classical = np.cov(train_features[mask2], rowvar=False)
print(f"Kovariansmatrise for blues: {C_blues}")
print(f"Kovariansmatrise for klassisk: {C_classical}")
C_blues_inv = np.linalg.inv(C_blues)
C_classical_inv = np.linalg.inv(C_classical)

correctly_classified = 0
i = 0


for test_feature in test_features_all:
    true_label = test_labels_all[i]
    diff_euc = train_features - test_feature
    euclidian_distances = np.linalg.norm(diff_euc, axis=1)
    idx = np.argpartition(euclidian_distances, k)[:k]
    nearest_labels = train_labels[idx]
    nearest_distances = euclidian_distances[idx]

    counts = Counter(nearest_labels)
    max_count = max(counts.values())

    candidates = [label for label, count in counts.items() if count == max_count]

    if len(candidates) == 1:
        majority_class = candidates[0]
    else:
        distance_sums = {
            label: np.sum(nearest_distances[nearest_labels == label])
            for label in candidates
        }
        majority_class = min(distance_sums, key=distance_sums.get)

    if majority_class == true_label:
        correctly_classified += 1

    i += 1

    diff_blues = test_feature - mu_blues
    diff_classical = test_feature - mu_classical
    mahalanobis_distance_to_blues = np.sqrt(diff_blues @ C_blues_inv @ diff_blues)
    mahalanobis_distance_to_classical = np.sqrt(diff_classical @ C_classical_inv @ diff_classical)
    if mahalanobis_distance_to_blues < mahalanobis_distance_to_classical:
        predicted_label = genre1
    else:
        predicted_label = genre2
    if predicted_label == true_label:
        correctly_classified += 1


treff_forhold = correctly_classified / len(test_features_all)
print(f"Treffprosent: {100*treff_forhold:.2f}%")


plt.figure()
plt.scatter(
    train_features[mask1, 0],
    train_features[mask1, 1],
    label=genre1
)
plt.scatter(
    train_features[mask2, 0],
    train_features[mask2, 1],
    label=genre2
)


filename = output_folder / "Scatter_2_features_blues_og_classical.png"
plt.legend()
plt.xlabel("spectral_rolloff_mean standardisert")
plt.ylabel("mfcc_1_mean standardisert")
plt.axis("equal")
plt.tight_layout()
plt.savefig(filename, dpi=300)
plt.close()