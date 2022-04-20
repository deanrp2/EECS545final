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
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.plots import plot_convergence
from skopt.utils import use_named_args

import tensorflow as tf

def errorChecking(y_target, y_preds):
    return mean_squared_error(y_target, y_preds)

def split(X, Y):
    Xtrain, Xtemp, Ytrain, Ytemp = train_test_split(X, Y, train_size = .5)
    Xtest, Xvalid, Ytest, Yvalid = train_test_split(Xtemp, Ytemp, train_size= .5)
    return Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest

opt = True

dnn_opt_calls = 50

opt = False
dnn_opt_calls = 11


def dnn(Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest):
    def make_model(Xtrain, Ytrain, hopts):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(hopts["width"], activation=hopts["activation"], input_dim=6))
        for _ in range(hopts["nlayers"]):
            #model.add(tf.keras.layers.Dense(hopts["width"], activation=tf.keras.layers.LeakyReLU(alpha=0.01)))
            model.add(tf.keras.layers.Dense(hopts["width"], activation = hopts["activation"]))
        model.add(tf.keras.layers.Dense(1, activation="relu"))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hopts["learning_rate"]),
                      loss="mean_squared_error")

        model.fit(Xtrain, Ytrain, epochs=hopts["epochs"], verbose=0)
        print(model.summary())
        return model

    def dnn_loss(Xtrain, Xvalid, Ytrain, Yvalid, hopts):
        model = make_model(Xtrain, Ytrain, hopts)
        Ypred = model.predict(Xvalid)
        a = errorChecking(Yvalid, Ypred)
        return a

    dim_width = Integer(low=1, high=int(60), name="width")
    dim_nlayers = Integer(low=1, high=int(20), name="nlayers")
    dim_activation = Categorical(["relu"], name="activation")
    dim_epochs = Integer(low=200, high=1500, name="epochs")
    dim_learning_rate = Categorical([0.01, 0.001, 0.0001], name="learning_rate")
    dims = [dim_width, dim_nlayers, dim_activation, dim_epochs, dim_learning_rate]

    @use_named_args(dimensions=dims)
    def fitness(width, nlayers, activation, epochs, learning_rate):
        hopts = {"width": width,
                 "nlayers": nlayers,
                 "activation": activation,
                 "epochs": epochs,
                 "learning_rate": learning_rate}
        return dnn_loss(Xtrain, Xvalid, Ytrain, Yvalid, hopts)

    # use this for relu
    initial_guess = [28, 5, "relu", 700, 0.001]
    # use this for leaky relu
    #initial_guess = [64, 2, "relu", 700, 0.001]

    if opt:
        search_result = gp_minimize(func=fitness, dimensions=dims, n_calls=dnn_opt_calls, x0=initial_guess,
                                    verbose=True)

        hopts = {"width": search_result.x[0],
                 "nlayers": search_result.x[1],
                 "activation": search_result.x[2],
                 "epochs": search_result.x[3],
                 "learning_rate": search_result.x[4]
                 }
    else:
        hopts = {"width": initial_guess[0],
                 "nlayers": initial_guess[1],
                 "activation": initial_guess[2],
                 "epochs": initial_guess[3],
                 "learning_rate": initial_guess[4]
                 }

    opt_model = make_model(Xtrain, Ytrain, hopts)
    opt_model.fit(Xtrain, Ytrain)

    # training losses
    Ypred = opt_model.predict(Xtrain)
    print(Ypred[:10])
    print(Ytrain[:10])

    train_losses = errorChecking(Ypred, Ytrain)

    # CV Losses
    Ypred = opt_model.predict(Xvalid)
    cvLosses = errorChecking(Ypred, Yvalid)

    # testing losses
    Ypred = opt_model.predict(Xtest)

    test_losses = errorChecking(Ypred, Ytest)

    return train_losses, cvLosses, test_losses, hopts

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

    # processing
    Y = Y.astype(float)

    # splitting the datasets
    Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest = split(X, Y)
    dsets = [Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest]

    train_losses, cvLosses, test_losses, hopts = dnn(Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest)
    print(train_losses)
    print(cvLosses)
    print(test_losses)
    print(hopts)

    # training
