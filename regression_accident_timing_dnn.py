import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


import tensorflow as tf

def errorChecking(y_target, y_preds):
    return mean_squared_error(y_target, y_preds)

def split(X, Y):
    Xtrain, Xtemp, Ytrain, Ytemp = train_test_split(X, Y, train_size = .5)
    Xtest, Xvalid, Ytest, Yvalid = train_test_split(Xtemp, Ytemp, train_size= .5)
    return Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest


if __name__ == "__main__":
    data = pd.read_csv("accident_classification/operation_accidents.csv")
    data.drop("accident", axis=1, inplace=True)
    data = data.loc[data['accident_time'] != 'None']
    data.reset_index(drop=True, inplace=True)

    # load x data
    rXdf = data.iloc[:, :6]
    rXnames = rXdf.columns
    rX = rXdf.values

    # preprocess x data and apply feature mappings
    scaler = MinMaxScaler()
    X = scaler.fit_transform(rX)

    # load y data and preprocessing
    rYdf = data.iloc[:, 6]
    rYnames = rYdf.index
    Y = rYdf.values

    #processing
    Y = Y.astype(float)

    # splitting the datasets
    Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest = split(X, Y)
    dsets = [Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest]

    print(1)

    # training
