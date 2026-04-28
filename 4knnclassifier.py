import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import Counter
from datautilities import readfromtabcsv, zscorenormalize
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

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

    # print("The classes of the 5 nearest neighbors:", labels[indexes_of_k_closest])
    # print("Guess:", majority_class)

    return majority_class


# kNN-modell
k = 7


# kNN k=5 classifier with normalized axes stepwise updated to include only the best features:
# 67.17%. Inferior to k=7 model
colsfork5 = [
    "mfcc_1_mean",
     "spectral_contrast_mean",
     "mfcc_5_std",
     "chroma_stft_9_mean",
     "rmse_mean",
     "rmse_var",
     "spectral_centroid_mean",
     "mfcc_6_mean",
     "spectral_contrast_var",
]
# For k=7 30s: 72.22%
# For k=7 10s: 63.63%
# For k=7 5s: 60.86%
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

duration = "30s"
train_features,train_labels,test_features,test_labels = readfromtabcsv("Music files/GenreClassData_"+duration+".txt",cols)

#Z score normalize axes
mean = np.mean(train_features,axis=0)
std = np.sqrt(np.var(train_features,axis=0))
train_features = (train_features-mean)/std
test_features = (test_features-mean)/std

genres = np.unique(train_labels)

genre_to_index = {genres[i]: i for i in range(len(genres))}
confusion_matrix = np.zeros((len(genres),len(genres)))
correct_per_genre = {g: 0 for g in genres}
total_per_genre = {g: 0 for g in genres}
correctly_classified = 0


for test_feature,true_label in zip(test_features,test_labels):

    majority_class = find_nearest(train_features,train_labels,test_feature)
    
    confusion_matrix[genre_to_index[true_label]][genre_to_index[majority_class]]+= 1

    if majority_class == true_label:
        correctly_classified += 1
        correct_per_genre[true_label] += 1

    total_per_genre[true_label] += 1




treff_forhold = correctly_classified / len(test_features)
print(f"Treffprosent totalt: {100*treff_forhold:.2f}%")


print("\nTreffprosent per sjanger:")

for g in genres:
    if total_per_genre[g] > 0:
        acc = correct_per_genre[g] / total_per_genre[g]
        print(f"{g}: {100*acc:.2f}% ({correct_per_genre[g]}/{total_per_genre[g]})")



disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                            display_labels=genres,
                            # cmap=plt.cm.Blues
                            )
# disp.plot()
disp.plot(cmap="Blues")
plt.xticks(rotation=45, ha="right")  # rotate x-axis labels
plt.title("Confusion matrix kNN classification, k="+str(k))
plt.tight_layout()
plt.savefig("plots/4knn"+duration+"confusion",bbox_inches="tight")
