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

print("sklearn")


scaler = StandardScaler()
features = scaler.fit_transform(features)


test_index = 35
test_features = features[test_index]
test_label = labels[test_index]

features = np.delete(features, (test_index), axis=0)
labels = np.delete(labels, (test_index), axis=0)

distances = np.linalg.norm(features - test_features, axis=1)
idx = np.argpartition(distances, k)[:k]
nearest_labels = labels[idx]
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


print("True label:", test_label)

print("The classes of the 5 nearest neighbors:", labels[idx])

print("Guess:", majority_class)