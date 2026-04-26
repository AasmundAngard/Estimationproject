# Design a classifier for all ten genres that classifies the audio tracks, each
# represented by a Track ID. You are allowed to use any classifier, as many
# features as you like and all of the available data sets GenreClassData 5s.txt,
# GenreClassData 10s.txt, and GenreClassData 30s.txt as input data.

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
    "Track ID",
    "spectral_rolloff_mean",
    "mfcc_1_mean",
    "spectral_centroid_mean",
    "tempo",
    "Genre"
]

df = pd.read_csv("Music files/GenreClassData_10s.txt", sep="\t", usecols=cols)

df_train = df.iloc[:2377]
df_test = df.iloc[2377:]

train_grouped = df_train.groupby("Track ID", as_index=False).mean(numeric_only=True)
train_genres = df_train.groupby("Track ID", as_index=False)["Genre"].first()
train_grouped = train_grouped.merge(train_genres, on="Track ID")

test_grouped = df_test.groupby("Track ID", as_index=False).mean(numeric_only=True)
test_genres = df_test.groupby("Track ID", as_index=False)["Genre"].first()
test_grouped = test_grouped.merge(test_genres, on="Track ID")

train_features = train_grouped.drop(columns=["Track ID", "Genre"]).values
train_labels = train_grouped["Genre"].values

test_features_all = test_grouped.drop(columns=["Track ID", "Genre"]).values
test_labels_all = test_grouped["Genre"].values

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features_all = scaler.transform(test_features_all)

# test_index = 35
# test_features = features[test_index]
# test_label = labels[test_index]

# features = np.delete(features, (test_index), axis=0)
# labels = np.delete(labels, (test_index), axis=0)

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

    i += 1


treff_forhold = correctly_classified / len(test_features_all)
print(f"Treffprosent: {100*treff_forhold:.2f}%")

print(Counter(train_labels))
print(Counter(test_labels_all))