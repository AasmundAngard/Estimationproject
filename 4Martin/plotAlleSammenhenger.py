import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
output_folder = BASE_DIR / "Boksplots" / "AlleSammenhenger"
output_folder.mkdir(parents=True, exist_ok=True)

data_file = BASE_DIR / "Music files" / "GenreClassData_30s.txt"
df_all = pd.read_csv(data_file, sep="\t")

# Kolonne 3 til og med 65 dersom vi teller kolonner fra 1.
# I Python betyr det indeks 2 til 64, altså [2:65].
features = list(df_all.columns[2:65])

# Henter alle sjangre automatisk
genres = sorted(df_all["Genre"].unique())

# Lager ett boxplot per feature
for feature in features:
    data = []
    labels = []

    for genre in genres:
        values = df_all.loc[df_all["Genre"] == genre, feature].dropna()
        data.append(values)
        labels.append(genre)

    plt.figure(figsize=(11, 6))
    plt.boxplot(data, labels=labels)
    plt.xlabel("Genre")
    plt.ylabel(feature)
    plt.title(f"Fordeling av {feature} for alle sjangre")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    safe_feature_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", feature)
    filename = f"{output_folder}/{safe_feature_name}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

print(f"Lagret {len(features)} plott i mappen: {output_folder}")




from sklearn.feature_selection import f_classif

X = df_all[features].values
y = df_all["Genre"].values

F, p = f_classif(X, y)

scores = dict(zip(features, F))

sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

for f, s in sorted_scores[:10]:
    print(f"{f}: {s:.2f}")