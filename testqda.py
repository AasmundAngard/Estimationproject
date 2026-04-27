# Design a Mahalanobis QDA (Quadratic Discriminant Analysis) algorithm
# to classify music genres.




import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import Counter


# For testing
def mahalanobis(means,covar,labels,test_features):
    # Mahalanobis-avstand:
    # D = sqrt( (x-mu).T dot C^-1 dot (cx-mu))
    D = [np.sqrt(np.dot(np.dot((test_features-means[g]).T,np.linalg.inv(covar[g])),(test_features-means[g])) ) for g in range(len(labels))]

    closest_label_index = np.argmin(D)
    guess = labels[closest_label_index]


    return guess



# Modellen trenes på alle parametre

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

datatypes = datatypes[:len(datatypes)//2]

cols = [
    "spectral_rolloff_mean",
    "spectral_centroid_mean",
    'spectral_rolloff_var', 
    'spectral_contrast_mean',
    "rmse_var",
    "mfcc_5_std" 
]

# cols = datatypes


df = pd.read_csv("Music files/GenreClassData_5s.txt", usecols=cols+["Genre","Type"], sep="\t")
labels = df["Genre"].values
genres = list(set(labels))


training_data = df[df["Type"]=="Train"]
testing_data = df[df["Type"]=="Test"]

training_features = training_data.drop(columns=["Genre","Type"]).values
training_labels = training_data["Genre"].values

test_features = testing_data.drop(columns=["Genre","Type"]).values
test_labels = testing_data["Genre"].values

masks = [training_labels==genre for genre in genres]



# Beregn mean for hver feature, for hver sjanger
# Beregn kovariansmatrise for hver sjanger

means = np.array([np.mean(training_features[mask],axis=0) for mask in masks])
covar = np.array([np.cov(training_features[mask],rowvar=False) for mask in masks])


right_guess = 0
wrong_guess = 0
confusion = np.zeros((len(genres),len(genres)))
for point in range(len(test_features)):
    true_class = test_labels[point]
    guess = mahalanobis(means,covar,genres,test_features[point])
    # print(true_class,guess)
    confusion[genres.index(true_class)][genres.index(guess)] += 1
    if guess == true_class:
        right_guess += 1
    else:
        wrong_guess += 1
print(genres)
print(confusion)
correctpercentage = right_guess/(right_guess+wrong_guess)
print("Andel riktig:",correctpercentage)
print("totalt gjett:",right_guess+wrong_guess)




# Beste var 53.03% med spectrall_rolloff_mean + spectral_centroid_mean  og  rmse_var + mfcc_5_std 

