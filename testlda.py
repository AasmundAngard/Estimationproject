import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import Counter
from datautilities import readfromtabcsv, zscorenormalize
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

k = 5

features = [
"mfcc_1_mean",                
"spectral_bandwidth_mean",    
"spectral_centroid_var",      
"spectral_bandwidth_mean",    
"spectral_centroid_var",      
"spectral_rolloff_mean",      
"mfcc_1_mean",                
"spectral_bandwidth_mean",    
"spectral_rolloff_mean",      
"mfcc_1_mean",                
"spectral_bandwidth_mean",    
"spectral_centroid_var",      
"spectral_bandwidth_mean",    
"spectral_rolloff_mean",      
"spectral_centroid_var",      
"spectral_bandwidth_mean",    
"spectral_centroid_var",      
"mfcc_1_mean",                
"mfcc_1_mean",                
"spectral_bandwidth_mean",    
"spectral_rolloff_mean",      
"mfcc_1_mean",                
"spectral_bandwidth_mean",    
"spectral_rolloff_mean",      
"mfcc_1_mean",                
"spectral_bandwidth_mean",    
"spectral_rolloff_mean",      
"mfcc_1_mean",                
"spectral_bandwidth_mean",    
"spectral_rolloff_mean",     
]

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

    return majority_class

# Programmet skal teste klassifisering med features fra 4findbestfeaturesvariance.py

cols = [
     "mfcc_1_mean",
]
cols = np.unique(features)

# Retrieve all features

df = pd.read_csv("Music files/GenreClassData_30s.txt", sep="\t")
features_30 = df[df["Type"]=="Train"][cols]
train_labels = df[df["Type"]=="Train"]["Genre"].values
test_30 = df[df["Type"]=="Test"][cols]
test_labels = df[df["Type"]=="Test"]["Genre"].values
df = pd.read_csv("Music files/GenreClassData_10s.txt", sep="\t")
features_10 = df[df["Type"]=="Train"][cols]
test_10 = df[df["Type"]=="Test"][cols]
df = pd.read_csv("Music files/GenreClassData_5s.txt", sep="\t")
features_5 = df[df["Type"]=="Train"][cols]
test_5 = df[df["Type"]=="Test"][cols]


# Collect training features
features_30 = features_30[features[:3]]

features_10_1 = features_10[features[3:6]].iloc[0::3].reset_index(drop=True)
features_10_2 = features_10[features[6:9]].iloc[1::3].reset_index(drop=True)
features_10_3 = features_10[features[9:12]].iloc[2::3].reset_index(drop=True)

features_5_1 = features_5[features[12:15]].iloc[0::6].reset_index(drop=True)
features_5_2 = features_5[features[15:18]].iloc[1::6].reset_index(drop=True)
features_5_3 = features_5[features[18:21]].iloc[2::6].reset_index(drop=True)
features_5_4 = features_5[features[21:24]].iloc[3::6].reset_index(drop=True)
features_5_5 = features_5[features[24:27]].iloc[4::6].reset_index(drop=True)
features_5_6 = features_5[features[27:30]].iloc[5::6].reset_index(drop=True)

# Collect test features
test_30 = test_30[features[:3]]

test_10_1 = test_10[features[3:6]].iloc[0::3].reset_index(drop=True)
test_10_2 = test_10[features[6:9]].iloc[1::3].reset_index(drop=True)
test_10_3 = test_10[features[9:12]].iloc[2::3].reset_index(drop=True)

test_5_1 = test_5[features[12:15]].iloc[0::6].reset_index(drop=True)
test_5_2 = test_5[features[15:18]].iloc[1::6].reset_index(drop=True)
test_5_3 = test_5[features[18:21]].iloc[2::6].reset_index(drop=True)
test_5_4 = test_5[features[21:24]].iloc[3::6].reset_index(drop=True)
test_5_5 = test_5[features[24:27]].iloc[4::6].reset_index(drop=True)
test_5_6 = test_5[features[27:30]].iloc[5::6].reset_index(drop=True)

train_features = np.hstack([
    features_30.values,
    features_10_1.values,
    features_10_2.values,
    features_10_3.values,
    features_5_1.values,
    features_5_2.values,
    features_5_3.values,
    features_5_4.values,
    features_5_5.values,
    features_5_6.values
])


test_features = np.hstack([
    test_30.values,
    test_10_1.values,
    test_10_2.values,
    test_10_3.values,
    test_5_1.values,
    test_5_2.values,
    test_5_3.values,
    test_5_4.values,
    test_5_5.values,
    test_5_6.values
])


means = np.mean(train_features,axis=0)
stds = np.sqrt(np.var(train_features,axis=0))

train_features = (train_features-means)/stds
test_features = (test_features-means)/stds


classes = np.unique(train_labels)

n_features = train_features.shape[1]

means_by_class = {}
priors = {}
covariance = np.zeros((n_features, n_features))

# Compute class means and priors
for c in classes:
    X_c = train_features[train_labels == c]
    means_by_class[c] = np.mean(X_c, axis=0)
    priors[c] = len(X_c) / len(train_features)

# Compute shared covariance matrix
for c in classes:
    X_c = train_features[train_labels == c]
    centered = X_c - means_by_class[c]
    covariance += centered.T @ centered

covariance /= len(train_features)

covariance += 1e-4 * np.eye(n_features)

inv_cov = np.linalg.pinv(covariance)  # safer than inverse

def lda_predict(x):
    scores = {}

    for c in classes:
        mu = means_by_class[c]

        term1 = x @ inv_cov @ mu
        term2 = -0.5 * mu @ inv_cov @ mu
        term3 = np.log(priors[c])

        scores[c] = term1 + term2 + term3

    return max(scores, key=scores.get)


predictions = np.array([lda_predict(x) for x in test_features])
print(predictions)
ConfusionMatrixDisplay.from_predictions(test_labels, predictions)
plt.show()