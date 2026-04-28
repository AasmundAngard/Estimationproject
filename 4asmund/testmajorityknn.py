# Mål: Design en modell som bruker 3 modeller til å stemme over valg av sjanger:
# 30s, 10s og 5s modell
# Kjøres på alle klippene, og stemmer

import numpy as np
from collections import Counter
import pandas as pd
# from sklearn.preprocessing import StandardScaler
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from datautilities import readfromtabcsv


# kNN-modell, k = 5
k = 5
def knnguess(model_features,model_labels,test_features,k=5):

    # Beregner avstand fra testpunkt til alle andre punkter
    distances = np.linalg.norm(model_features - test_features, axis=1)

    # Legger indeksene til de k punktene med kortest avstand først i arrayet, og henter dem ut
    idx_of_nearest_points = np.argpartition(distances, k)[:k]

    nearest_labels = model_labels[idx_of_nearest_points]
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

    return majority_class


cols = [
    "spectral_rolloff_mean",
    "mfcc_1_mean",
    "spectral_centroid_mean",
    # "tempo",
    "spectral_rolloff_var", # Det beste for å bytte ut tempo
    # "rmse_var", # Det beste istedenfor tempo om akser normaliseres
]
# Lag tre modeller
clip_durations = ["5s","10s","30s"]
filepath = "Music files/GenreClassData_"
num_recordings = 990


for duration in clip_durations:
    train_features,train_labels,test_features,test_labels = readfromtabcsv(filepath+duration+".txt",cols)
    # Normaliser aksene til mu=0, std=1
    train_means = np.mean(train_features, axis=0)
    train_stds = np.sqrt(np.var(train_features, axis=0))
    train_features -= train_means
    train_features /= train_stds

    test_features -= train_means
    test_features /= train_stds

    match duration:
        case "30s":
            model_features30s = train_features
            model_labels30s = train_labels
            test_features30s = test_features
            test_labels30s = test_labels
        case "10s":
            model_features10s = train_features
            model_labels10s = train_labels
            # Hvert opptak har en matrise med feature vektorer:
            test_features10s = test_features.reshape(-1,3,test_features.shape[1])
            test_labels10s = test_labels.reshape(-1,3)
            # test_features10s = np.array([test_features[i::3] for i in range(3)])
            # test_labels10s = np.array([test_labels[i::3] for i in range(3)])
        case "5s":
            model_features5s = train_features
            model_labels5s = train_labels
            # Hvert opptak har en matrise med feature vektorer:
            test_features5s = test_features.reshape(-1,6,test_features.shape[1])
            test_labels5s = test_labels.reshape(-1,6)
            # test_features5s = np.array([test_features[i::6] for i in range(6)])
            # test_labels5s = np.array([test_labels[i::6] for i in range(6)])


genres = np.unique(train_labels)

# genre_to_index = {genres[i]: i for i in range(len(genres))}
# correct_per_genre = {g: 0 for g in genres}
# total_per_genre = {g: 0 for g in genres}
confusion_matrix = np.zeros((len(genres),len(genres)))
correctly_classified = 0


for point in range(len(test_features30s)):
    votes = np.zeros(genres.shape[0])
    guess30s = knnguess(model_features30s,model_labels30s,test_features30s[point])
    guess10s = [knnguess(model_features10s,model_labels10s,test_features) for test_features in test_features10s[point]]
    guess5s = [knnguess(model_features5s,model_labels5s,test_features) for test_features in test_features5s[point]]
    
    all_votes = [guess30s, Counter(guess10s).most_common(1)[0][0], Counter(guess5s).most_common(1)[0][0]]

    final_guess = Counter(all_votes).most_common(1)[0][0]
    print(test_labels30s[point],final_guess)

    if test_labels30s[point] == final_guess:
        correctly_classified += 1





treff_forhold = correctly_classified / len(test_features30s)
print(f"Treffprosent totalt: {100*treff_forhold:.2f}%")


print("\nTreffprosent per sjanger:")

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
plt.title("Confusion matrix kNN classification, k=5")
plt.tight_layout()
plt.savefig("plots/abrakadabra")