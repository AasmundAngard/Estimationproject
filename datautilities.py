import pandas as pd
import numpy as np


def readfromtabcsv(filepath,cols):
    df = pd.read_csv(filepath, sep="\t", usecols=cols+["Genre","Type"])

    train_features = df[df["Type"]=="Train"][cols].values
    train_labels = df[df["Type"]=="Train"]["Genre"].values

    test_features = df[df["Type"]=="Test"][cols].values
    test_labels = df[df["Type"]=="Test"]["Genre"].values

    return train_features,train_labels,test_features,test_labels

def zscorenormalize(matrix,mean=None,std=None):
    # Expects matrix on form (samples,features)
    # vector = (vector-mu)/sigma
    if mean!=None and std!=None:
        normalized_matrix = (matrix-mean)/std
    else:
        mean = np.mean(matrix,axis=0)
        std = np.sqrt(np.var(matrix,axis=0))
        normalized_matrix = (matrix-mean)/std

    return normalized_matrix, mean, std

