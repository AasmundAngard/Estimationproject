import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import Counter
from datautilities import readfromtabcsv, zscorenormalize
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# kNN modell med en kombinasjon av variasjon mellom features mellom 5s klipp og features i 30s klipp

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


# kNN-modell
k = 7

cols = [
     "spectral_flatness_var",
     "spectral_contrast_var",
     "mfcc_8_std",
     "rmse_var",
     "chroma_stft_4_mean",
     "spectral_bandwidth_mean",
     "mfcc_5_std",
     "spectral_flatness_mean",
     "spectral_rolloff_mean",
     "spectral_contrast_mean",
     "mfcc_3_std",
     "rmse_mean",
     "mfcc_6_mean",
     "spectral_rolloff_var",
     "spectral_flatness_var",
     "mfcc_2_mean",
     "spectral_contrast_var",
]
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
    "mfcc_1_mean",
    # "tempo"
     "rmse_var",
]

train_features_5,train_labels_5,test_features_5,test_labels_5 = readfromtabcsv("Music files/GenreClassData_"+"5s"+".txt",cols)
train_features_10,train_labels_10,test_features_10,test_labels_10 = readfromtabcsv("Music files/GenreClassData_"+"10s"+".txt",cols)
train_features_30,train_labels_30,test_features_30,test_labels_30 = readfromtabcsv("Music files/GenreClassData_"+"30s"+".txt",cols)

#Z score normalize axes
mean = np.mean(train_features_30,axis=0)
std = np.sqrt(np.var(train_features_30,axis=0))
train_features_30 = (train_features_30-mean)/std
test_features_30 = (test_features_30-mean)/std

#Z score normalize axes
mean = np.mean(train_features_10,axis=0)
std = np.sqrt(np.var(train_features_10,axis=0))
train_features_10 = (train_features_10-mean)/std
test_features_10 = (test_features_10-mean)/std

#Z score normalize axes
mean = np.mean(train_features_5,axis=0)
std = np.sqrt(np.var(train_features_5,axis=0))
train_features_5 = (train_features_5-mean)/std
test_features_5 = (test_features_5-mean)/std

genres = np.unique(train_labels_5)

# Split into one matrix for each song (every 3 contigous clips together)
train_features_10 = train_features_10.reshape(-1, 3, train_features_10.shape[1])
test_features_10 = test_features_10.reshape(-1, 3, test_features_10.shape[1])

train_labels_10 = train_labels_10.reshape(-1,3)[:, 0]
test_labels_10 = test_labels_10.reshape(-1,3)[:, 0]

# Split into one matrix for each song (every 6 contigous clips together)
train_features_5 = train_features_5.reshape(-1, 6, train_features_5.shape[1])
test_features_5 = test_features_5.reshape(-1, 6, test_features_5.shape[1])

train_labels_5 = train_labels_5.reshape(-1,6)[:, 0]
test_labels_5 = test_labels_5.reshape(-1,6)[:, 0]

# Calculate variance across clips for every song
train_vars_10 = np.var(train_features_10, axis=1)
test_vars_10 = np.var(test_features_10,axis=1)

train_vars_5 = np.var(train_features_5, axis=1)
test_vars_5 = np.var(test_features_5,axis=1)

train_combined = np.concatenate([train_features_30, train_vars_10, train_vars_5], axis=1)
test_combined = np.concatenate([test_features_30, test_vars_10, test_vars_5], axis=1)

print(train_combined)

genre_to_index = {genres[i]: i for i in range(len(genres))}
confusion_matrix = np.zeros((len(genres),len(genres)))
correct_per_genre = {g: 0 for g in genres}
total_per_genre = {g: 0 for g in genres}
correctly_classified = 0

correct = 0
total = 0
for test_var,true_label in zip(test_combined,test_labels_30):
    
    majority_class = find_nearest(train_combined,train_labels_30,test_var)
    print(true_label,majority_class)
    if majority_class == true_label:
        correct += 1

    confusion_matrix[genre_to_index[true_label]][genre_to_index[majority_class]]+= 1

    total += 1
    # correct_per_genre[true_label] += 1

    # total_per_genre[true_label] += 1

print(correct,"/",total)
print(correct/total)







# for test_feature,true_label in zip(test_features,test_labels):

#     majority_class = find_nearest(train_features,train_labels,test_feature)
    
#     confusion_matrix[genre_to_index[true_label]][genre_to_index[majority_class]]+= 1

#     if majority_class == true_label:
#         correctly_classified += 1





# treff_forhold = correctly_classified / len(test_features)
# print(f"Treffprosent totalt: {100*treff_forhold:.2f}%")


# print("\nTreffprosent per sjanger:")

# for g in genres:
#     if total_per_genre[g] > 0:
#         acc = correct_per_genre[g] / total_per_genre[g]
#         print(f"{g}: {100*acc:.2f}% ({correct_per_genre[g]}/{total_per_genre[g]})")



disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                            display_labels=genres,
                            # cmap=plt.cm.Blues
                            )
# disp.plot()
disp.plot(cmap="Blues")
plt.xticks(rotation=45, ha="right")  # rotate x-axis labels
plt.title("Confusion matrix kNN classification, k="+str(k))
plt.tight_layout()
plt.savefig("plots/selectfeatvar5var10feat30",bbox_inches="tight")
