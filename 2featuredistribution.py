import pandas as pd
import matplotlib.pyplot as plt

features = ["spectral_rolloff_mean",
            "mfcc_1_mean",
            "spectral_centroid_mean",
            "tempo"]

df_all = pd.read_csv("Music files/GenreClassData_30s.txt", sep="\t")
cols = features + ["Genre"]
df = df_all[cols]

genres = ["pop", "disco", "metal", "classical"]

for feature in features:
    data = []
    for genre in genres:
        values = df[df["Genre"] == genre][feature]
        data.append(values)
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=genres)
    plt.xlabel("Genre")
    plt.ylabel(feature)
    plt.title(f"Fordeling av {feature} for ulike sjangre")
    filename = f"boksplots/{feature}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()