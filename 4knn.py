import numpy as np
import pandas as pd
from collections import Counter
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# kNN-modell
k = 10

# For testing
def find_nearest(model_features,model_labels,test_features):

    test_features = test_features.reshape(1,-1)

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

# With top 10 variance scores
cols = ['mfcc_1_mean_10_2', 'spectral_bandwidth_mean_30', 'spectral_centroid_var_30', 
        'spectral_rolloff_mean_30', 'spectral_centroid_mean_30', 'zero_cross_rate_std_30', 
        'mfcc_4_mean_30', 'rmse_var_5_4', 'mfcc_2_mean_30', 'spectral_flatness_var_30', 
        'spectral_rolloff_var_5_2', 'rmse_mean_10_3', 'spectral_contrast_var_5_2'
        ]

lengths = ["30","10_1","10_2","10_3","5_1","5_2","5_3","5_4","5_5","5_6"]
cols = [datatype +"_"+ suffix for datatype in datatypes for suffix in lengths]

# With top 15 variance scores
# cols = ['mfcc_1_mean_10_2', 'spectral_bandwidth_mean_30', 'spectral_centroid_var_30', 
#         'spectral_rolloff_mean_30', 'spectral_centroid_mean_30', 'zero_cross_rate_std_30', 
#         'mfcc_4_mean_30', 'rmse_var_5_4', 'mfcc_2_mean_30', 'spectral_flatness_var_30', 
#         'spectral_rolloff_var_5_2', 'rmse_mean_10_3', 'spectral_contrast_var_30', 'chroma_stft_2_mean_30', 
#         'chroma_stft_7_mean_30', 'mfcc_4_std_5_2', 'mfcc_5_std_5_2', 'mfcc_6_std_5_5', 'mfcc_8_mean_10_3', 
#         'spectral_flatness_mean_10_3', 'mfcc_6_mean_5_3']
# With top 20 variance scores
# cols = ['mfcc_1_mean_10_2', 'spectral_bandwidth_mean_30', 'spectral_centroid_var_30', 'spectral_rolloff_mean_30', 'spectral_centroid_mean_30', 'zero_cross_rate_std_30', 'mfcc_4_mean_30', 'rmse_var_5_4', 'mfcc_2_mean_30', 'spectral_flatness_var_30', 'spectral_rolloff_var_5_2', 'rmse_mean_10_3', 'spectral_contrast_var_30', 'chroma_stft_2_mean_30', 'chroma_stft_7_mean_30', 'chroma_stft_9_mean_30', 'mfcc_8_mean_30', 'mfcc_6_mean_30', 'chroma_stft_5_mean_30', 'spectral_flatness_mean_10_3', 'mfcc_4_std_5_2', 'mfcc_5_std_5_2', 'mfcc_6_std_5_5', 'spectral_contrast_mean_10_1', 'mfcc_9_mean_10_2', 'mfcc_7_std_5_2']
# With top 30 variance scores
# cols = ['mfcc_1_mean_10_2', 'spectral_bandwidth_mean_30', 'spectral_centroid_var_30', 'spectral_rolloff_mean_30', 'spectral_centroid_mean_30', 'zero_cross_rate_std_30', 'mfcc_4_mean_30', 'rmse_var_5_4', 'mfcc_2_mean_30', 'spectral_flatness_var_30', 'spectral_rolloff_var_5_2', 'rmse_mean_10_3', 'spectral_contrast_var_30', 'chroma_stft_2_mean_30', 'chroma_stft_7_mean_30', 'chroma_stft_9_mean_30', 'mfcc_8_mean_30', 'mfcc_6_mean_30', 'chroma_stft_5_mean_30', 'spectral_flatness_mean_10_3', 'chroma_stft_4_mean_30', 'chroma_stft_12_mean_30', 'mfcc_6_std_5_5', 'mfcc_5_std_5_2', 'spectral_contrast_mean_30', 'mfcc_4_std_5_2', 'mfcc_9_mean_30', 'zero_cross_rate_mean_30', 'mfcc_7_std_10_1', 'mfcc_12_mean_30', 'mfcc_7_mean_5_4', 'spectral_bandwidth_var_5_2', 'mfcc_3_std_5_6', 'mfcc_3_mean_5_2']

# Follows same order as datatypes, but shows correctpercentage
correctpercentage = np.zeros(0)

# Retrieve all features

df = pd.read_csv("Music files/GenreClassData_30s.txt", sep="\t")
features_30 = df[df["Type"]=="Train"][datatypes]
test_30 = df[df["Type"]=="Test"][datatypes]
train_labels = df[df["Type"]=="Train"]["Genre"].values
test_labels = df[df["Type"]=="Test"]["Genre"].values

df = pd.read_csv("Music files/GenreClassData_10s.txt", sep="\t")
features_10 = df[df["Type"]=="Train"][datatypes]
test_10 = df[df["Type"]=="Test"][datatypes]
df = pd.read_csv("Music files/GenreClassData_5s.txt", sep="\t")
features_5 = df[df["Type"]=="Train"][datatypes]
test_5 = df[df["Type"]=="Test"][datatypes]


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

test_30 = test_30.reset_index(drop=True)
test_10_1 = test_10.iloc[0::3].reset_index(drop=True)
test_10_2 = test_10.iloc[1::3].reset_index(drop=True)
test_10_3 = test_10.iloc[2::3].reset_index(drop=True)

test_5_1 = test_5.iloc[0::6].reset_index(drop=True)
test_5_2 = test_5.iloc[1::6].reset_index(drop=True)
test_5_3 = test_5.iloc[2::6].reset_index(drop=True)
test_5_4 = test_5.iloc[3::6].reset_index(drop=True)
test_5_5 = test_5.iloc[4::6].reset_index(drop=True)
test_5_6 = test_5.iloc[5::6].reset_index(drop=True)


combined_test = pd.concat([
    test_30.add_suffix("_30"),

    test_10_1.add_suffix("_10_1"),
    test_10_2.add_suffix("_10_2"),
    test_10_3.add_suffix("_10_3"),

    test_5_1.add_suffix("_5_1"),
    test_5_2.add_suffix("_5_2"),
    test_5_3.add_suffix("_5_3"),
    test_5_4.add_suffix("_5_4"),
    test_5_5.add_suffix("_5_5"),
    test_5_6.add_suffix("_5_6")
], axis=1)

print(combined_test)


datatypes = combined_features.columns.tolist()

test_df = combined_test
train_df = combined_features



# Normaliser aksene til mu=0, std=1
model_means = np.mean(train_df, axis=0)
model_stds = np.sqrt(np.var(train_df, axis=0))
train_df -= model_means
train_df /= model_stds

test_df -= model_means
test_df /= model_stds

train_features = train_df[cols].values
test_features = test_df[cols].values

genres = np.unique(train_labels)
genre_to_index = {genres[i]: i for i in range(len(genres))}
confusion_matrix = np.zeros((len(genres),len(genres)))
correct = 0
for test_feature, true_label in zip(test_features,test_labels):
    guess_label = find_nearest(train_features,train_labels,test_feature)
    print(true_label,guess_label)
    if true_label == guess_label:
        correct += 1
    confusion_matrix[genre_to_index[true_label]][genre_to_index[guess_label]] += 1

print(correct/len(test_labels))

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                            display_labels=genres,
                            # cmap=plt.cm.Blues
                            )
# disp.plot()
disp.plot(cmap="Blues")
plt.xticks(rotation=45, ha="right")  # rotate x-axis labels
plt.title("Confusion matrix kNN classification, k="+str(k))
plt.tight_layout()
plt.savefig("plots/4knnwithallmaxvarianceconfusion",bbox_inches="tight")

print(len(cols))