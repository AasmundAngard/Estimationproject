# Choose features by testing features one by one from all available features



import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import Counter

# kNN-modell
k = 10

# For testing
def find_nearest(model_features,model_labels,test_features):

    distances = np.linalg.norm(model_features - test_features, axis=1)
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


    return majority_class


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
    "spectral_flatness_mean_5_5", #31%
    "rmse_var_10_3", # 41%
    "spectral_contrast_var_30", # 47%
    "mfcc_9_mean_10_1", # 53%
    "spectral_bandwidth_mean_30", # 57%
    "mfcc_5_std_10_1", # 61%
    "spectral_contrast_var_5_5", # 65%
    "spectral_contrast_var_5_5", # 66%
]

# Follows same order as datatypes, but shows correctpercentage
correctpercentage = np.zeros(0)

# Retrieve all features

df = pd.read_csv("Music files/GenreClassData_30s.txt", sep="\t")
features_30 = df[df["Type"]=="Train"][datatypes]
labels = df["Genre"].values
df = pd.read_csv("Music files/GenreClassData_10s.txt", sep="\t")
features_10 = df[df["Type"]=="Train"][datatypes]
df = pd.read_csv("Music files/GenreClassData_5s.txt", sep="\t")
features_5 = df[df["Type"]=="Train"][datatypes]


features_10_1 = features_10.iloc[0::3].reset_index(drop=True)
features_10_2 = features_10.iloc[1::3].reset_index(drop=True)
features_10_3 = features_10.iloc[2::3].reset_index(drop=True)

features_5_1 = features_5.iloc[0::6].reset_index(drop=True)
features_5_2 = features_5.iloc[1::6].reset_index(drop=True)
features_5_3 = features_5.iloc[2::6].reset_index(drop=True)
features_5_4 = features_5.iloc[3::6].reset_index(drop=True)
features_5_5 = features_5.iloc[4::6].reset_index(drop=True)
features_5_6 = features_5.iloc[5::6].reset_index(drop=True)

combined_features = pd.concat([
    features_30.add_suffix("_30"),

    features_10_1.add_suffix("_10_1"),
    features_10_2.add_suffix("_10_2"),
    features_10_3.add_suffix("_10_3"),

    features_5_1.add_suffix("_5_1"),
    features_5_2.add_suffix("_5_2"),
    features_5_3.add_suffix("_5_3"),
    features_5_4.add_suffix("_5_4"),
    features_5_5.add_suffix("_5_5"),
    features_5_6.add_suffix("_5_6")
], axis=1)

datatypes = combined_features.columns.tolist()

test_df = combined_features.iloc[3::4].reset_index(drop=True)
train_df = combined_features.drop(combined_features.index[3::4]).reset_index(drop=True)

test_labels = labels[3::4]
train_labels = np.delete(labels, np.arange(3, len(labels), 4))


# Normaliser aksene til mu=0, std=1
model_means = np.mean(train_df, axis=0)
model_stds = np.sqrt(np.var(train_df, axis=0))
train_df -= model_means
train_df /= model_stds

test_df -= model_means
test_df /= model_stds




for feature in datatypes:

        model_features = train_df[cols+[feature]].values
        test_features = test_df[cols+[feature]].values

        right_guess = 0
        wrong_guess = 0
        for point in range(len(test_features)):
            true_class = test_labels[point]
            guess = find_nearest(model_features,train_labels,test_features[point])
            # print(true_class,guess)
            if guess == true_class:
                right_guess += 1
            else:
                wrong_guess += 1

        percentage = right_guess/(right_guess+wrong_guess)
        correctpercentage = np.append(correctpercentage,percentage)
        # print(feature1,feature2,percentage)
        print(feature,percentage)


best_index = np.argmax(correctpercentage)


print("Beste:",datatypes[best_index],correctpercentage[best_index])

