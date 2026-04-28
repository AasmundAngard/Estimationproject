from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from datautilities import readfromtabcsv

# X = feature matrix (n_samples × n_features)
# y = labels
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
train_features,train_labels,test_features,test_labels = readfromtabcsv("Music files/GenreClassData_30s.txt",datatypes)


model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)
print(train_features.shape)
print(train_labels.shape)
model.fit(train_features, train_labels)

correct = 0
guess = 0
for feature,label in zip(test_features,test_labels):
    predictions = model.predict([feature])
    if predictions == label:
        correct+= 1
    guess += 1

print(correct,"/",guess)
print(correct/guess)
