import numpy as np
import pandas as pd
from collections import Counter
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# With top 10 variance scores
# cols = ['mfcc_1_mean_10_2', 'spectral_bandwidth_mean_30', 'spectral_centroid_var_30', 
#         'spectral_rolloff_mean_30', 'spectral_centroid_mean_30', 'zero_cross_rate_std_30', 
#         'mfcc_4_mean_30', 'rmse_var_5_4', 'mfcc_2_mean_30', 'spectral_flatness_var_30', 
#         'spectral_rolloff_var_5_2', 'rmse_mean_10_3', 'spectral_contrast_var_5_2'
#         ]

# lengths = ["30","10_1","10_2","10_3","5_1","5_2","5_3","5_4","5_5","5_6"]
# cols = [datatype +"_"+ suffix for datatype in datatypes for suffix in lengths]

# With top 15 variance scores
# cols = ['mfcc_1_mean_10_2', 'spectral_bandwidth_mean_30', 'spectral_centroid_var_30', 
#         'spectral_rolloff_mean_30', 'spectral_centroid_mean_30', 'zero_cross_rate_std_30', 
#         'mfcc_4_mean_30', 'rmse_var_5_4', 'mfcc_2_mean_30', 'spectral_flatness_var_30', 
#         'spectral_rolloff_var_5_2', 'rmse_mean_10_3', 'spectral_contrast_var_30', 'chroma_stft_2_mean_30', 
#         'chroma_stft_7_mean_30', 'mfcc_4_std_5_2', 'mfcc_5_std_5_2', 'mfcc_6_std_5_5', 'mfcc_8_mean_10_3', 
#         'spectral_flatness_mean_10_3', 'mfcc_6_mean_5_3']

# With top 20 variance scores
cols = ['mfcc_1_mean_10_2', 'spectral_bandwidth_mean_30', 'spectral_centroid_var_30', 'spectral_rolloff_mean_30', 'spectral_centroid_mean_30', 'zero_cross_rate_std_30', 'mfcc_4_mean_30', 'rmse_var_5_4', 'mfcc_2_mean_30', 'spectral_flatness_var_30', 'spectral_rolloff_var_5_2', 'rmse_mean_10_3', 'spectral_contrast_var_30', 'chroma_stft_2_mean_30', 'chroma_stft_7_mean_30', 'chroma_stft_9_mean_30', 'mfcc_8_mean_30', 'mfcc_6_mean_30', 'chroma_stft_5_mean_30', 'spectral_flatness_mean_10_3', 'mfcc_4_std_5_2', 'mfcc_5_std_5_2', 'mfcc_6_std_5_5', 'spectral_contrast_mean_10_1', 'mfcc_9_mean_10_2', 'mfcc_7_std_5_2']
# With top 30 variance scores
# cols = ['mfcc_1_mean_10_2', 'spectral_bandwidth_mean_30', 'spectral_centroid_var_30', 'spectral_rolloff_mean_30', 'spectral_centroid_mean_30', 'zero_cross_rate_std_30', 'mfcc_4_mean_30', 'rmse_var_5_4', 'mfcc_2_mean_30', 'spectral_flatness_var_30', 'spectral_rolloff_var_5_2', 'rmse_mean_10_3', 'spectral_contrast_var_30', 'chroma_stft_2_mean_30', 'chroma_stft_7_mean_30', 'chroma_stft_9_mean_30', 'mfcc_8_mean_30', 'mfcc_6_mean_30', 'chroma_stft_5_mean_30', 'spectral_flatness_mean_10_3', 'chroma_stft_4_mean_30', 'chroma_stft_12_mean_30', 'mfcc_6_std_5_5', 'mfcc_5_std_5_2', 'spectral_contrast_mean_30', 'mfcc_4_std_5_2', 'mfcc_9_mean_30', 'zero_cross_rate_mean_30', 'mfcc_7_std_10_1', 'mfcc_12_mean_30', 'mfcc_7_mean_5_4', 'spectral_bandwidth_var_5_2', 'mfcc_3_std_5_6', 'mfcc_3_mean_5_2']


# Follows same order as datatypes, but shows correctpercentage
correctpercentage = np.zeros(0)

# Retrieve all features

df = pd.read_csv("Music files/GenreClassData_30s.txt", sep="\t")
features_30 = df[df["Type"]=="Train"]
test_30 = df[df["Type"]=="Test"]
train_labels = df[df["Type"]=="Train"]["Genre"].values
test_labels = df[df["Type"]=="Test"]["Genre"].values

df = pd.read_csv("Music files/GenreClassData_10s.txt", sep="\t")
features_10 = df[df["Type"]=="Train"]
test_10 = df[df["Type"]=="Test"]
df = pd.read_csv("Music files/GenreClassData_5s.txt", sep="\t")
features_5 = df[df["Type"]=="Train"]
test_5 = df[df["Type"]=="Test"]


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

test_features = combined_test[cols].values
train_features = combined_features[cols].values

# Normaliser aksene til mu=0, std=1
model_means = np.mean(train_features, axis=0)
model_stds = np.sqrt(np.var(train_features, axis=0))
train_features -= model_means
train_features /= model_stds

test_features -= model_means
test_features /= model_stds



classes = np.unique(train_labels)

means = {}
priors = {}

for c in classes:
    X_c = train_features[train_labels == c]

    means[c] = np.mean(X_c, axis=0)
    priors[c] = len(X_c) / len(train_features)


cov = np.cov(train_features, rowvar=False)

cov_inv = np.linalg.inv(cov)

def lda_predict(x):

    scores = {}

    for c in classes:

        mu = means[c]

        # for LDA: g_i = mu @ C^-1 @ x + log(Pr(w_i)) - 0.5 mu @ C^-1 @ mu
        score = np.log(priors[c]) + x @ cov_inv @ mu - 0.5 * mu @ cov_inv @ mu

        scores[c] = score

    return max(scores, key=scores.get)

genres = np.unique(train_labels)
genre_to_index = {genres[i]: i for i in range(len(genres))}
confusion_matrix = np.zeros((len(genres),len(genres)))
correct = 0
for test_feature, true_label in zip(test_features,test_labels):
    guess_label = lda_predict(test_feature)
    # print(true_label,guess_label)
    if true_label == guess_label:
        correct += 1
    confusion_matrix[genre_to_index[true_label]][genre_to_index[guess_label]] += 1

print("Recall rate:",correct/len(test_labels))

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                            display_labels=genres,
                            # cmap=plt.cm.Blues
                            )
# disp.plot()
disp.plot(cmap="Blues")
plt.xticks(rotation=45, ha="right")  # rotate x-axis labels
plt.title("Confusion matrix LDA classification")
plt.tight_layout()
plt.savefig("plots/4ldawith20maxvarianceconfusion",bbox_inches="tight")

print("num features:",len(cols))