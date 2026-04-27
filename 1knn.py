import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# kNN-modell, k = 5
k = 5


# Modellen skal trenes på parameterne spectral_rolloff_mean (index 10), mfcc_1_mean (index 41)
# spectral_centroid_mean (index 6), tempo (index 40) GenreID (index 65)


cols = [
    "spectral_rolloff_mean",
    "mfcc_1_mean",
    "spectral_centroid_mean",
    "tempo",
    # "rmse_var",
    "Genre",
    "Type"
]

df = pd.read_csv("Music files/GenreClassData_30s.txt", sep="\t", usecols=cols)

train_features = df[df["Type"]=="Train"].drop(columns=["Genre","Type"]).values
train_labels = df[df["Type"]=="Train"]["Genre"].values

test_features = df[df["Type"]=="Test"].drop(columns=["Genre","Type"]).values
test_labels = df[df["Type"]=="Test"]["Genre"].values

genres = np.unique(train_labels)

genre_to_index = {genres[i]: i for i in range(len(genres))}
confusion_matrix = np.zeros((len(genres),len(genres)))
correct_per_genre = {g: 0 for g in genres}
total_per_genre = {g: 0 for g in genres}
correctly_classified = 0


for test_feature,true_label in zip(test_features,test_labels):

    # Beregner avstand fra testpunkt til alle andre punkter
    distances = np.linalg.norm(train_features - test_feature, axis=1)

    # Legger indexene til de k punktene med kortest avstand først i arrayet, og henter dem ut
    idx_of_nearest_points = np.argpartition(distances, k)[:k]

    nearest_labels = train_labels[idx_of_nearest_points]
    nearest_distances = distances[idx_of_nearest_points]

    counts = Counter(nearest_labels)
    max_count = max(counts.values())

    # Labels som har flest punkter blant k nærmeste
    candidates = [label for label, count in counts.items() if count == max_count]

    if len(candidates) == 1:
        majority_class = candidates[0]
    else:
        distance_sums = {
            label: np.sum(nearest_distances[nearest_labels == label])
            for label in candidates
        }
        majority_class = min(distance_sums, key=distance_sums.get)
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

disp.plot(cmap="Blues")
plt.xticks(rotation=45, ha="right")  # rotate x-axis labels
plt.title("Confusion matrix kNN classification, k=5")
plt.tight_layout()
plt.savefig("plots/1knnconfusion")