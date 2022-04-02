import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import time
from classify_accident_classification_sk import *

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.plots import plot_convergence
from skopt.utils import use_named_args

import tensorflow as tf

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

opt = True
dnn_opt_calls = 50

def dnn(Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest, Xtrans, Ytrans):
    def make_model(Xtrain, Ytrain, hopts):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(hopts["width"], activation = hopts["activation"], input_dim = 6))
        for _ in range(hopts["nlayers"]):
            model.add(tf.keras.layers.Dense(hopts["width"], activation = hopts["activation"]))
        model.add(tf.keras.layers.Dense(4, activation = "softmax"))

        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = hopts["learning_rate"]),
                loss = "categorical_crossentropy")

        model.fit(Xtrain, Ytrain, epochs = hopts["epochs"], verbose = 0)
        print(model.summary())
        return model

    def dnn_loss(Xtrain, Xvalid, Ytrain, Yvalid, hopts):
        model = make_model(Xtrain, Ytrain, hopts)
        Ypred_prob = model.predict(Xvalid)
        Ypred = Ypred_prob.argmax(1)
        a = loss(Ypred, Ypred_prob, Yvalid)
        return a

    dim_width = Integer(low = 1, high = int(60), name = "width")
    dim_nlayers= Integer(low = 1, high = int(10), name = "nlayers")
    #dim_activation= Categorical(["relu", "sigmoid", "softmax", "softplus",
    #    "softsign", "tanh", "selu", "elu", "exponential"], name = "activation")
    dim_activation= Categorical(["sigmoid"], name = "activation")
    dim_epochs = Integer(low = 200, high = 400, name = "epochs")
    dim_learning_rate = Categorical([0.01, 0.001, 0.0001], name = "learning_rate")
    dims = [dim_width, dim_nlayers, dim_activation, dim_epochs, dim_learning_rate]

    @use_named_args(dimensions = dims)
    def fitness(width, nlayers, activation, epochs, learning_rate):
        hopts = {"width" : width,
                "nlayers" : nlayers,
                "activation" : activation, 
                "epochs" : epochs, 
                "learning_rate" : learning_rate}
        return -dnn_loss(Xtrain, Xvalid, Ytrain, Yvalid, hopts)

    initial_guess = [39, 4, "sigmoid", 400, 0.01]

    if opt:
        search_result = gp_minimize(func = fitness, dimensions = dims, n_calls = dnn_opt_calls,
                x0 = initial_guess, verbose = True)

        hopts = {"width" : search_result.x[0],
                 "nlayers" : search_result.x[1],
                 "activation" : search_result.x[2],
                 "epochs" : search_result.x[3],
                 "learning_rate": search_result.x[4]
                 }
    else:
        hopts = {"width" : initial_guess[0],
                 "nlayers" : initial_guess[1],
                 "activation" : initial_guess[2],
                 "epochs" : initial_guess[3],
                 "learning_rate" : initial_guess[4]
                 }


    opt_model = make_model(Xtrain, Ytrain, hopts)
    opt_model.fit(Xtrain, Ytrain)

    #training losses
    Ypred_prob = opt_model.predict(Xtrain)
    Ypred = Ypred_prob.argmax(1)

    train_losses = all_loss(Ypred, Ypred_prob, Ytrain)

    #testing losses
    Ypred_prob = opt_model.predict(Xtest)
    Ypred = Ypred_prob.argmax(1)

    test_losses = all_loss(Ypred, Ypred_prob, Ytest)

    return train_losses, test_losses, hopts

if __name__ == "__main__":
    start = time.time()

    #import data
    data = pd.read_csv("accident_classification/operation_accidents.csv")

    # load x data
    rXdf = data.iloc[:,:6]
    rXnames = rXdf.columns
    rX = rXdf.values

    # preprocess x data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(rX)

    # load y data
    rYdf = data.iloc[:, 6]
    rYnames = rYdf.index
    rY = rYdf.values

    # preprocess y data
    #label_encoder = LabelEncoder()
    #Y = label_encoder.fit_transform(rY)
    onehot_encoder = OneHotEncoder()
    Y = onehot_encoder.fit_transform(rY.reshape(-1, 1))
    Y = Y.toarray()

    # train, test, verification split
    Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest  = split(X, Y)
    dsets = [Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest]

    rs = [dnn(*dsets, scaler, onehot_encoder)]

    cols = ["Method"] + ["train_" + a for a in metric_opts]+ ["test_" + a for a in metric_opts]
    cols = [a.ljust(15) for a in cols]
    names = "dnn"
    for i, r in enumerate(rs):
        pt = ""
        pt += names[i].ljust(15)
        for m in metric_opts:
            pt += ("%.4f"%r[0][m]).ljust(15)
        for m in metric_opts:
            pt += ("%.4f"%r[1][m]).ljust(15)
        print(pt)

    print("Best hopts\n", rs[0][2])



    print("Rutime:", time.time() - start, "s")

