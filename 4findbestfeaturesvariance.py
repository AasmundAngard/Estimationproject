import numpy as np
import pandas as pd



# Programmet skal beregne mean for alle sjangre i alle 630 features, 
# og så sjekke varians på tvers av sjangre for alle features.
# Deretter sjekke den gjennomsnittlige variansen innen hver sjanger, for hver feature

# Deretter velge ut features med høyest:
# varians mellom sjangre / varians innen sjangre


# Test alle features, finn beste.

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
     
]

# Retrieve all features

df = pd.read_csv("Music files/GenreClassData_30s.txt", sep="\t")
features_30 = df[df["Type"]=="Train"][["Genre"]+datatypes]
labels = df["Genre"].values
df = pd.read_csv("Music files/GenreClassData_10s.txt", sep="\t")
features_10 = df[df["Type"]=="Train"][["Genre"]+datatypes]
df = pd.read_csv("Music files/GenreClassData_5s.txt", sep="\t")
features_5 = df[df["Type"]=="Train"][["Genre"]+datatypes]


features_10_1 = features_10.iloc[0::3].reset_index(drop=True)
features_10_2 = features_10.iloc[1::3].reset_index(drop=True)
features_10_3 = features_10.iloc[2::3].reset_index(drop=True)

features_5_1 = features_5.iloc[0::6].reset_index(drop=True)
features_5_2 = features_5.iloc[1::6].reset_index(drop=True)
features_5_3 = features_5.iloc[2::6].reset_index(drop=True)
features_5_4 = features_5.iloc[3::6].reset_index(drop=True)
features_5_5 = features_5.iloc[4::6].reset_index(drop=True)
features_5_6 = features_5.iloc[5::6].reset_index(drop=True)


# Calculate the genre means for each feature
genre_means_30 = features_30.groupby("Genre")[datatypes].mean()

genre_means_10_1 = features_10_1.groupby("Genre")[datatypes].mean()
genre_means_10_2 = features_10_2.groupby("Genre")[datatypes].mean()
genre_means_10_3 = features_10_3.groupby("Genre")[datatypes].mean()

genre_means_5_1 = features_5_1.groupby("Genre")[datatypes].mean()
genre_means_5_2 = features_5_2.groupby("Genre")[datatypes].mean()
genre_means_5_3 = features_5_3.groupby("Genre")[datatypes].mean()
genre_means_5_4 = features_5_4.groupby("Genre")[datatypes].mean()
genre_means_5_5 = features_5_5.groupby("Genre")[datatypes].mean()
genre_means_5_6 = features_5_6.groupby("Genre")[datatypes].mean()

# Calculate the variance in mean between genres for each feature
variance_between_classes_30 = genre_means_30.var(axis=0)

variance_between_classes_10_1 = genre_means_10_1.var(axis=0)
variance_between_classes_10_2 = genre_means_10_2.var(axis=0)
variance_between_classes_10_3 = genre_means_10_3.var(axis=0)

variance_between_classes_5_1 = genre_means_5_1.var(axis=0)
variance_between_classes_5_2 = genre_means_5_2.var(axis=0)
variance_between_classes_5_3 = genre_means_5_3.var(axis=0)
variance_between_classes_5_4 = genre_means_5_4.var(axis=0)
variance_between_classes_5_5 = genre_means_5_5.var(axis=0)
variance_between_classes_5_6 = genre_means_5_6.var(axis=0)


# Calculate the average variance within each genre for each feature
variance_within_class_30 = features_30.groupby("Genre")[datatypes].var().mean(axis=0)

variance_within_class_10_1 = features_10_1.groupby("Genre")[datatypes].var().mean(axis=0)
variance_within_class_10_2 = features_10_2.groupby("Genre")[datatypes].var().mean(axis=0)
variance_within_class_10_3 = features_10_3.groupby("Genre")[datatypes].var().mean(axis=0)

variance_within_class_5_1 = features_5_1.groupby("Genre")[datatypes].var().mean(axis=0)
variance_within_class_5_2 = features_5_2.groupby("Genre")[datatypes].var().mean(axis=0)
variance_within_class_5_3 = features_5_3.groupby("Genre")[datatypes].var().mean(axis=0)
variance_within_class_5_4 = features_5_4.groupby("Genre")[datatypes].var().mean(axis=0)
variance_within_class_5_5 = features_5_5.groupby("Genre")[datatypes].var().mean(axis=0)
variance_within_class_5_6 = features_5_6.groupby("Genre")[datatypes].var().mean(axis=0)

# Calculate the relation between: variance between genres / variance within genres
variance_relation_30 = variance_between_classes_30 / variance_within_class_30

variance_relation_10_1 = variance_between_classes_10_1 / variance_within_class_10_1
variance_relation_10_2 = variance_between_classes_10_2 / variance_within_class_10_2
variance_relation_10_3 = variance_between_classes_10_3 / variance_within_class_10_3

variance_relation_5_1 = variance_between_classes_5_1 / variance_within_class_5_1
variance_relation_5_2 = variance_between_classes_5_2 / variance_within_class_5_2
variance_relation_5_3 = variance_between_classes_5_3 / variance_within_class_5_3
variance_relation_5_4 = variance_between_classes_5_4 / variance_within_class_5_4
variance_relation_5_5 = variance_between_classes_5_5 / variance_within_class_5_5
variance_relation_5_6 = variance_between_classes_5_6 / variance_within_class_5_6


# Find features with highest relation

highest = 20
variance_relation_30 = variance_relation_30.sort_values(ascending=False).head(highest)
print(variance_relation_30)

variance_relation_10_1 = variance_relation_10_1.sort_values(ascending=False).head(highest)
variance_relation_10_2 = variance_relation_10_2.sort_values(ascending=False).head(highest)
variance_relation_10_3 = variance_relation_10_3.sort_values(ascending=False).head(highest)
print(variance_relation_10_1)
print(variance_relation_10_2)
print(variance_relation_10_3)

variance_relation_5_1 = variance_relation_5_1.sort_values(ascending=False).head(highest)
variance_relation_5_2 = variance_relation_5_2.sort_values(ascending=False).head(highest)
variance_relation_5_3 = variance_relation_5_3.sort_values(ascending=False).head(highest)
variance_relation_5_4 = variance_relation_5_4.sort_values(ascending=False).head(highest)
variance_relation_5_5 = variance_relation_5_5.sort_values(ascending=False).head(highest)
variance_relation_5_6 = variance_relation_5_6.sort_values(ascending=False).head(highest)
print(variance_relation_5_1)
print(variance_relation_5_2)
print(variance_relation_5_3)
print(variance_relation_5_4)
print(variance_relation_5_5)
print(variance_relation_5_6)

# Start med en feature, og finn korrelasjon itl alle andre
feature_scores = pd.concat([
    variance_relation_30.rename("30"),
    variance_relation_10_1.rename("10_1"),
    variance_relation_10_2.rename("10_2"),
    variance_relation_10_3.rename("10_3"),
    variance_relation_5_1.rename("5_1"),
    variance_relation_5_2.rename("5_2"),
    variance_relation_5_3.rename("5_3"),
    variance_relation_5_4.rename("5_4"),
    variance_relation_5_5.rename("5_5"),
    variance_relation_5_6.rename("5_6")
], axis=1)

print(feature_scores)

best_columns = feature_scores.idxmax(axis=1)
print(best_columns)
formatted_list = [
    str(feature)+"_"+str(col) for feature, col in best_columns.items()
]
print(formatted_list)

