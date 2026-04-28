import pandas as pd


def readfromtabcsv(filepath,cols):
    df = pd.read_csv(filepath, sep="\t", usecols=cols+["Genre","Type"])

    train_features = df[df["Type"]=="Train"].drop(columns=["Genre","Type"]).values
    train_labels = df[df["Type"]=="Train"]["Genre"].values

    test_features = df[df["Type"]=="Test"].drop(columns=["Genre","Type"]).values
    test_labels = df[df["Type"]=="Test"]["Genre"].values

    return train_features,train_labels,test_features,test_labels