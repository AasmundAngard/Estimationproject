import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import Counter

# kNN-modell, k = 5
k = 5

# Modellen skal trenes på parameterne spectral_rolloff_mean (index 10), mfcc_1_mean (index 41)
# spectral_centroid_mean (index 6), tempo (index 40) GenreID (index 65)


# Shape: 990


cols = [
    "spectral_rolloff_mean",
    "mfcc_1_mean",
    "spectral_centroid_mean",
    "tempo",
    "Genre"
]

df = pd.read_csv("Music files/GenreClassData_30s.txt", sep="\t", usecols=cols)
features = df.drop(columns=["Genre"]).values
labels = df["Genre"].values
train_features = features[:794]
train_labels = labels[:794]
genres = np.unique(train_labels)

test_features_all = features[794:]
test_labels_all = labels[794:]

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features_all = scaler.transform(test_features_all)


correct_per_genre = {g: 0 for g in genres}
total_per_genre = {g: 0 for g in genres}
correctly_classified = 0
i = 0

for test_feature in test_features_all:
    true_label = test_labels_all[i]
    distances = np.linalg.norm(train_features - test_feature, axis=1)
    idx = np.argpartition(distances, k)[:k]
    nearest_labels = train_labels[idx]
    nearest_distances = distances[idx]

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
        correct_per_genre[true_label] += 1

    total_per_genre[true_label] += 1

    i += 1


treff_forhold = correctly_classified / len(test_features_all)
print(f"Treffprosent totalt: {100*treff_forhold:.2f}%")


print("\nTreffprosent per sjanger:")

for g in genres:
    if total_per_genre[g] > 0:
        acc = correct_per_genre[g] / total_per_genre[g]
        print(f"{g}: {100*acc:.2f}% ({correct_per_genre[g]}/{total_per_genre[g]})")