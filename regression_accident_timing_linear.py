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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

#import tensorflow as tf

def errorChecking(y_target, y_preds):
    return mean_squared_error(y_target, y_preds)

def split(X, Y):
    Xtrain, Xtemp, Ytrain, Ytemp = train_test_split(X, Y, train_size = .5)
    Xtest, Xvalid, Ytest, Yvalid = train_test_split(Xtemp, Ytemp, train_size= .5)
    return Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest

def multinomial(X, deg):
    currPow = X
    out = X
    for i in range(1,deg):
        currPow = np.multiply(currPow, X)
        out = np.concatenate((out, currPow), axis=1)

    return out



def linearRegression(XTrain, YTrain, Xvalid, Yvalid, Xtest, Ytest):
    model = LinearRegression()
    model.fit(XTrain, YTrain)

    # Training Losses
    Y_pred = model.predict(XTrain)
    training_loss = errorChecking(YTrain, Y_pred)

    # Cross validation Losses
    Y_pred = model.predict(Xvalid)
    valid_loss = errorChecking(Yvalid, Y_pred)


    # Testing Losses
    Y_pred = model.predict(Xtest)
    test_loss = errorChecking(Ytest, Y_pred)

    params = model.get_params(deep=True)

    return {"training loss": training_loss, "validation loss": valid_loss, "test loss": test_loss, "params":params}

def ridgeRegression(XTrain, YTrain, Xvalid, Yvalid, Xtest, Ytest, regConst):
    model = Ridge(alpha=regConst)
    model.fit(XTrain, YTrain)

    # Training Losses
    Y_pred = model.predict(XTrain)
    training_loss = errorChecking(YTrain, Y_pred)

    # Cross validation Losses
    Y_pred = model.predict(Xvalid)
    valid_loss = errorChecking(Yvalid, Y_pred)

    # Testing Losses
    Y_pred = model.predict(Xtest)
    test_loss = errorChecking(Ytest, Y_pred)

    params = model.get_params(deep=True)

    return {"training loss": training_loss, "validation loss": valid_loss, "test loss": test_loss, "params": params}


def lassoRegression(XTrain, YTrain, Xvalid, Yvalid, Xtest, Ytest, regConst):
    model = Lasso(alpha=regConst, max_iter=10000)
    model.fit(XTrain, YTrain)

    # Training Losses
    Y_pred = model.predict(XTrain)
    training_loss = errorChecking(YTrain, Y_pred)

    # Cross validation Losses
    Y_pred = model.predict(Xvalid)
    valid_loss = errorChecking(Yvalid, Y_pred)

    # Testing Losses
    Y_pred = model.predict(Xtest)
    test_loss = errorChecking(Ytest, Y_pred)

    params = model.get_params(deep=True)

    return {"training loss": training_loss, "validation loss": valid_loss, "test loss": test_loss, "params": params}


def elasticNet(XTrain, YTrain, Xvalid, Yvalid, Xtest, Ytest, regConst,l1ratio):
    model = ElasticNet(alpha=regConst, l1_ratio=l1ratio,max_iter=10000)
    model.fit(XTrain, YTrain)

    # Training Losses
    Y_pred = model.predict(XTrain)
    training_loss = errorChecking(YTrain, Y_pred)

    # Cross validation Losses
    Y_pred = model.predict(Xvalid)
    valid_loss = errorChecking(Yvalid, Y_pred)

    # Testing Losses
    Y_pred = model.predict(Xtest)
    test_loss = errorChecking(Ytest, Y_pred)

    params = model.get_params(deep=True)

    return {"training loss": training_loss, "validation loss": valid_loss, "test loss": test_loss, "params": params}


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
    rX = multinomial(rX, 10)
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
    print(Xtrain.shape, Xtest.shape, Ytrain.shape)
    dsets = [Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest]

    # training

    linear = linearRegression(Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest)
    ridge = ridgeRegression(Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest, 1)
    lasso = lassoRegression(Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest, 1)
    elastic = elasticNet(Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest, 1, 0.25)
    print(linear)
    print(ridge)
    print(lasso)
    print(elastic)

    #training with feature mapping post normalization

    # for i in range(1,10):
    #     XtrainMap = multinomial(Xtrain, i)
    #     XvalidMap = multinomial(Xvalid, i)
    #     XtestMap = multinomial(Xtest, i)
    #     print(XtrainMap.shape)
    #     linear = linearRegression(XtrainMap, Ytrain, XvalidMap, Yvalid, XtestMap, Ytest)
    #     ridge = ridgeRegression(XtrainMap, Ytrain, XvalidMap, Yvalid, XtestMap, Ytest, 1)
    #     lasso = lassoRegression(XtrainMap, Ytrain, XvalidMap, Yvalid, XtestMap, Ytest, 1)
    #     elastic = elasticNet(XtrainMap, Ytrain, XvalidMap, Yvalid, XtestMap, Ytest, 1, 0.5)
    #     print(i)
    #     print(linear)
    #     print(ridge)
    #     print(lasso)
    #     print(elastic)


