import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
output_folder = BASE_DIR / "4Martin"

# cols = [
#     "spectral_rolloff_mean",
#     "mfcc_1_mean",
#     "spectral_centroid_mean",
#     "spectral_rolloff_var",
#     "Genre"
# ]

cols = [
    "mfcc_1_mean",
    "spectral_bandwidth_mean",
    "spectral_centroid_var",
    "spectral_rolloff_mean",
    "spectral_centroid_mean",
    "rmse_var",
    "mfcc_2_mean",
    "zero_cross_rate_std",
    "mfcc_4_mean",
    "spectral_contrast_var",
    "Genre"
]


data_file = BASE_DIR / "Music files" / "GenreClassData_30s.txt"
df = pd.read_csv(data_file, sep="\t", usecols=cols)
features = df.drop(columns=["Genre"]).values
labels = df["Genre"].values
train_features_values = features[:794]
print(f"train_features_values.shape: {train_features_values.shape}")
train_labels = labels[:794]
test_features_values_all = features[794:]
test_labels_all = labels[794:]

scaler = StandardScaler()
train_features_values = scaler.fit_transform(train_features_values)
test_features_values_all = scaler.transform(test_features_values_all)


genres = np.unique(train_labels)
means = {}
cov_invs = {}

for g in genres:
    Xc = train_features_values[train_labels == g]

    means[g] = np.mean(Xc, axis=0)

    C = np.cov(Xc, rowvar=False)
    cov_invs[g] = np.linalg.inv(C)


correct_per_genre = {g: 0 for g in genres}
total_per_genre = {g: 0 for g in genres}

correctly_classified = 0
i = 0
predicted_labels = []

for test_feature in test_features_values_all:
    true_label = test_labels_all[i]

    distances = {}

    for g in genres:
        diff = test_feature - means[g]
        d2 = diff @ cov_invs[g] @ diff
        distances[g] = d2

    predicted_label = min(distances, key=distances.get)
    predicted_labels.append(predicted_label)

    total_per_genre[true_label] += 1

    if predicted_label == true_label:
        correctly_classified += 1
        correct_per_genre[true_label] += 1

    i += 1


treff_forhold = correctly_classified / len(test_features_values_all)
print(f"Treffprosent: {100*treff_forhold:.2f}%")

print("\nTreffprosent per sjanger:")

for g in genres:
    if total_per_genre[g] > 0:
        acc = correct_per_genre[g] / total_per_genre[g]
        print(f"{g}: {100*acc:.2f}% ({correct_per_genre[g]}/{total_per_genre[g]})")

cm = confusion_matrix(test_labels_all, predicted_labels, labels=genres)

plt.figure(figsize=(9, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=genres)
disp.plot(cmap="Blues", xticks_rotation=45)

plt.title("Confusion matrix")
plt.tight_layout()
filename = output_folder / "Confusion_matrix_mahalanobis.png"
plt.savefig(filename, dpi=300)