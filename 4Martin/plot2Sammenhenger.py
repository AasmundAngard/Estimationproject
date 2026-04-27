import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
output_folder = BASE_DIR / "Boksplots" / "blues_vs_country"
output_folder.mkdir(parents=True, exist_ok=True)

# Velg sjangre her
genre1 = "country"
genre2 = "blues"

df_all = pd.read_csv("Music files/GenreClassData_30s.txt", sep="\t")

features = list(df_all.columns[2:65])
genres = [genre1, genre2]

for feature in features:
    data = []
    labels = []

    for genre in genres:
        values = df_all.loc[df_all["Genre"] == genre, feature].dropna()
        data.append(values)
        labels.append(genre)

    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=labels)
    plt.xlabel("Genre")
    plt.ylabel(feature)
    plt.title(f"Fordeling av {feature}: {genre1} vs {genre2}")
    plt.tight_layout()

    safe_feature_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", feature)
    filename = f"{output_folder}/{safe_feature_name}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

print(f"Lagret {len(features)} plott i mappen: {output_folder}")


scores = {}

for feature in features:
    x1 = df_all[df_all["Genre"] == "country"][feature].dropna()
    x2 = df_all[df_all["Genre"] == "blues"][feature].dropna()

    mu1, mu2 = np.mean(x1), np.mean(x2)
    var1, var2 = np.var(x1), np.var(x2)

    score = (mu1 - mu2)**2 / (var1 + var2)
    scores[feature] = score

# sorter
sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

for f, s in sorted_scores[:10]:
    print(f"{f}: {s:.4f}")