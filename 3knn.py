# Design a k-NN classifier (k=5) for all ten genres using only four features
# with at least three features being; spectral rolloff mean, mfcc 1 mean, 
# spectral centroid mean or tempo. Motivate why you selected the particular
# four features.



import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import Counter

# kNN-modell, k = 5
k = 5

# For testing
def find_nearest(model_features,model_labels,test_features,normalize_mean,normalize_std):
    normalized_test_features = (test_features-normalize_mean)/normalize_std

    distances = np.linalg.norm(model_features - normalized_test_features, axis=1)
    indexes_of_k_closest = np.argpartition(distances, k)[:k]
    nearest_labels = model_labels[indexes_of_k_closest]
    nearest_distances = distances[indexes_of_k_closest]


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

    # print("The classes of the 5 nearest neighbors:", labels[indexes_of_k_closest])
    # print("Guess:", majority_class)

    return majority_class






# ANTALL SAMPLES:
# Totalt 990 datapunkter
# 792 samples for modell
# 990 - 792 = 198 samples for testing
model_datapoints = 792


# Modellen skal trenes på 4 parametre, der minst 3 er en av:
# spectral_rolloff_mean (index 10), 
# mfcc_1_mean (index 41)
# spectral_centroid_mean (index 6), 
# tempo (index 40) GenreID (index 65)

datatypes = ['zero_cross_rate_mean', 'zero_cross_rate_std', 'rmse_mean', 
              'rmse_var', 'spectral_centroid_mean', 'spectral_centroid_var', 
              'spectral_bandwidth_mean', 'spectral_bandwidth_var', 
              'spectral_rolloff_mean', 'spectral_rolloff_var', 
              'spectral_contrast_mean', 'spectral_contrast_var', 
              'spectral_flatness_mean', 'spectral_flatness_var', 
              'chroma_stft_1_mean', 'chroma_stft_2_mean', 'chroma_stft_3_mean', 
              'chroma_stft_4_mean', 'chroma_stft_5_mean', 'chroma_stft_6_mean', 
              'chroma_stft_7_mean', 'chroma_stft_8_mean', 'chroma_stft_9_mean', 
              'chroma_stft_10_mean', 'chroma_stft_11_mean', 'chroma_stft_12_mean', 
              'chroma_stft_1_std', 'chroma_stft_2_std', 'chroma_stft_3_std', 
              'chroma_stft_4_std', 'chroma_stft_5_std', 'chroma_stft_6_std', 
              'chroma_stft_7_std', 'chroma_stft_8_std', 'chroma_stft_9_std', 
              'chroma_stft_10_std', 'chroma_stft_11_std', 'chroma_stft_12_std', 
              'tempo', 'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean', 'mfcc_4_mean', 
              'mfcc_5_mean', 'mfcc_6_mean', 'mfcc_7_mean', 'mfcc_8_mean', 
              'mfcc_9_mean', 'mfcc_10_mean', 'mfcc_11_mean', 'mfcc_12_mean', 
              'mfcc_1_std', 'mfcc_2_std', 'mfcc_3_std', 'mfcc_4_std', 'mfcc_5_std', 
              'mfcc_6_std', 'mfcc_7_std', 'mfcc_8_std', 'mfcc_9_std', 'mfcc_10_std', 
              'mfcc_11_std', 'mfcc_12_std'
              ]

cols = [
    "spectral_rolloff_mean",
    "spectral_centroid_mean",
    # "mfcc_1_mean",
    # "tempo"
]
# beste: rmse_var mfcc_5_std

# Follows same order as datatypes, but shows correctpercentage
correctpercentage = np.zeros(0)

# df = pd.read_csv("Music files/GenreClassData_30s.txt", sep="\t", usecols=cols)
df = pd.read_csv("Music files/GenreClassData_30s.txt", sep="\t")
# features = df.drop(columns=["Genre"]).values
labels = df["Genre"].values

for feature1 in datatypes:
    for feature2 in datatypes:
        features = df[cols+[feature1,feature2]].values

        model_features = features[:model_datapoints]
        model_labels = labels[:model_datapoints]

        test_features = features[model_datapoints:]
        test_labels = labels[model_datapoints:]


        # Normaliser aksene til mu=0, std=1
        model_means = np.mean(model_features, axis=0)
        model_stds = np.sqrt(np.var(model_features, axis=0))
        model_features -= model_means
        model_features /= model_stds

        right_guess = 0
        wrong_guess = 0
        for point in range(len(test_features)):
            true_class = test_labels[point]
            guess = find_nearest(model_features,model_labels,test_features[point],model_means,model_stds)
            # print(true_class,guess)
            if guess == true_class:
                right_guess += 1
            else:
                wrong_guess += 1

        percentage = right_guess/(right_guess+wrong_guess)
        correctpercentage = np.append(correctpercentage,percentage)
        print(feature1,feature2,percentage)


best_index1 = np.argmax(correctpercentage) // len(datatypes)
best_index2 = np.argmax(correctpercentage) % len(datatypes)

best_type1 = datatypes[best_index1]
best_type2 = datatypes[best_index2]
print("beste type1,2:",best_type1,best_type2,correctpercentage[best_index1*len(datatypes)+best_index2])

# Beste med spectrall_rolloff_mean+spectral_centroid_mean: rmse_var mfcc_5_std


