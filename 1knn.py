import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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


test_index = 100
test_features = features[test_index]
test_label = labels[test_index]

features = np.delete(features, (test_index), axis=0)
labels = np.delete(labels, (test_index), axis=0)

distances = np.linalg.norm(features - test_features, axis=1)

idx = np.argpartition(distances, k)[:k]

print("true label:", test_label)

print("guess:", labels[idx])

print("guess:", labels[idx])