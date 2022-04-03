def warn(*args, **kwargs):
        pass
import warnings
warnings.warn = warn

import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.plots import plot_convergence
from skopt.utils import use_named_args

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

import tensorflow as tf



metric_opts = ["acc", "bal"] #https://scikit-learn.org/stable/modules/model_evaluation.html
hyperopt_metric = "bal"

def loss(Ypred, Ypred_prob, Ytruei, metric = None): 
    if len(Ytruei.shape) == 2:
        Ytrue = Ytruei.argmax(1)
    else:
        Ytrue = Ytruei

    if metric == None:
        metric = hyperopt_metric

    if metric == "acc":
        return accuracy_score(Ytrue, Ypred)
    elif metric == "bal":
        return balanced_accuracy_score(Ytrue, Ypred)

def all_loss(Ypred, Ypred_prob, Ytrue):
    losses = {}
    for m in metric_opts:
        losses[m] = loss(Ypred, Ypred_prob, Ytrue, metric = m)
    return losses

def split(X, Y):
    Xtrain, Xtemp, Ytrain, Ytemp = train_test_split(X, Y, train_size = .5)
    Xtest, Xvalid, Ytest, Yvalid = train_test_split(Xtemp, Ytemp, train_size= .5)
    return Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest

#fxn takes training, verification, testing
#    returns training loss, testing loss & hyperparameters in dict

# knn
def knn(Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest, Xtrans, Ytrans):
    hopts = {"n_neighbors" : 22}

    opt_model = KNeighborsClassifier(**hopts)
    opt_model.fit(Xtrain, Ytrain)

    #training losses
    Ypred = opt_model.predict(Xtrain)
    Ypred_prob = opt_model.predict_proba(Xtrain)

    train_losses = all_loss(Ypred, Ypred_prob, Ytrain)

    #testing losses
    Ypred = opt_model.predict(Xtest)
    Ypred_prob = opt_model.predict_proba(Xtest)

    test_losses = all_loss(Ypred, Ypred_prob, Ytest)
    iYtest = Ytrans.inverse_transform(Ytest)
    iYpred = Ytrans.inverse_transform(Ypred)

    return train_losses, test_losses, hopts

# logistic regression
def logreg(Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest, Xtrans, Ytrans):
    opt_model = LogisticRegression()
    opt_model.fit(Xtrain, Ytrain)

    #training losses
    Ypred = opt_model.predict(Xtrain)
    Ypred_prob = opt_model.predict_proba(Xtrain)

    train_losses = all_loss(Ypred, Ypred_prob, Ytrain)

    #testing losses
    Ypred = opt_model.predict(Xtest)
    Ypred_prob = opt_model.predict_proba(Xtest)

    test_losses = all_loss(Ypred, Ypred_prob, Ytest)

    return train_losses, test_losses

def lda(Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest, Xtrans, Ytrans):
    opt_model = LinearDiscriminantAnalysis()
    opt_model.fit(Xtrain, Ytrain)

    #training losses
    Ypred = opt_model.predict(Xtrain)
    Ypred_prob = opt_model.predict_proba(Xtrain)

    train_losses = all_loss(Ypred, Ypred_prob, Ytrain)

    #testing losses
    Ypred = opt_model.predict(Xtest)
    Ypred_prob = opt_model.predict_proba(Xtest)

    test_losses = all_loss(Ypred, Ypred_prob, Ytest)

    return train_losses, test_losses


# support vector machine
def svm(Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest, Xtrans, Ytrans):
    hopts = {"C" : 3.54,
             "kernel" : "poly"}

    opt_model = SVC(probability = True, **hopts)
    opt_model.fit(Xtrain, Ytrain)

    #training losses
    Ypred = opt_model.predict(Xtrain)
    Ypred_prob = opt_model.predict_proba(Xtrain)

    train_losses = all_loss(Ypred, Ypred_prob, Ytrain)

    #testing losses
    Ypred = opt_model.predict(Xtest)
    Ypred_prob = opt_model.predict_proba(Xtest)

    test_losses = all_loss(Ypred, Ypred_prob, Ytest)

    return train_losses, test_losses, hopts

# decision tree
def tree(Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest, Xtrans, Ytrans):
    hopts = {"criterion" : "gini",
             "splitter" : "random",
             "max_depth" : 28}

    opt_model = DecisionTreeClassifier(**hopts)
    opt_model.fit(Xtrain, Ytrain)

    #training losses
    Ypred = opt_model.predict(Xtrain)
    Ypred_prob = opt_model.predict_proba(Xtrain)

    train_losses = all_loss(Ypred, Ypred_prob, Ytrain)

    #testing losses
    Ypred = opt_model.predict(Xtest)
    Ypred_prob = opt_model.predict_proba(Xtest)

    test_losses = all_loss(Ypred, Ypred_prob, Ytest)

    return train_losses, test_losses, hopts

# gaussian process
def gpc(Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest, Xtrans, Ytrans):
    def gpc_loss(Xtrain, Xvalid, Ytrain, Yvalid, hopts):
        model = GaussianProcessClassifier(**hopts)
        model.fit(Xtrain, Ytrain)
        Ypred = model.predict(Xvalid)
        Ypred_prob = model.predict_proba(Xvalid)

        return loss(Ypred, Ypred_prob, Yvalid)

    hopts = {}
    opt_model = GaussianProcessClassifier(**hopts)
    opt_model.fit(Xtrain, Ytrain)

    #training losses
    Ypred = opt_model.predict(Xtrain)
    Ypred_prob = opt_model.predict_proba(Xtrain)

    train_losses = all_loss(Ypred, Ypred_prob, Ytrain)

    #testing losses
    Ypred = opt_model.predict(Xtest)
    Ypred_prob = opt_model.predict_proba(Xtest)

    test_losses = all_loss(Ypred, Ypred_prob, Ytest)

    return train_losses, test_losses

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
        return model

    def dnn_loss(Xtrain, Xvalid, Ytrain, Yvalid, hopts):
        model = make_model(Xtrain, Ytrain, hopts)
        Ypred_prob = model.predict(Xvalid)
        Ypred = Ypred_prob.argmax(1)
        a = loss(Ypred, Ypred_prob, Yvalid)
        return a

    hopts = {"width" : 56,
             "nlayers" : 2,
             "activation" : "sigmoid",
             "epochs" : 432,
             "learning_rate" : 0.001
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


def BAS(n):
    assert n <= 40000
    #import data
    data = pd.read_csv("accident_classification/operation_accidents.csv").iloc[:n]

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
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(rY)
    hotencoder = OneHotEncoder()
    Yh = hotencoder.fit_transform(rY.reshape(-1, 1))
    Yh = Yh.toarray()

    # train, test, verification split
    Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest  = split(X, Y)
    dsets = [Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest]
    Xtrainh, Xvalidh, Xtesth, Ytrainh, Yvalidh, Ytesth  = split(X, Yh)
    dsetsh = [Xtrainh, Xvalidh, Xtesth, Ytrainh, Yvalidh, Ytesth]

    rs = [knn(*dsets, scaler, label_encoder),
    logreg(*dsets, scaler, label_encoder),
    lda(*dsets, scaler, label_encoder),
    svm(*dsets, scaler, label_encoder),
    tree(*dsets, scaler, label_encoder),
    gpc(*dsets, scaler, label_encoder),
    dnn(*dsetsh, scaler, hotencoder)]

    names = ["knn", "log regr", "lda", "svm", "tree", "gpc", "dnn"]

    rdict = {}
    for i, r in enumerate(rs):
        rdict[names[i]] = r[1]["bal"]

    return rdict




if __name__ == "__main__":
    N = 2


    a = 10
    names = ["knn", "log regr", "lda", "svm", "tree", "gpc", "dnn"]

    logpath = Path("dcomplexity.log")
    if not logpath.exists():
        with open(logpath, "w") as f:
            for n in names:
                f.write((n + ",").ljust(a))
            f.write("size")
            f.write("\n")

    def write_out(outdict, sz):
        with open(logpath, "a") as f:
            for n in names:
                f.write(("%.4f,"%outdict[n]).ljust(a))
            f.write(str(sz) + "\n")

    def loguniform(low=0, high=1, size=None):
        return np.power(10, np.random.uniform(low, high, size))

    b = loguniform(2.3, 4.4, N)
    for bb in b:
        print("Running b is", int(bb))
        intb = int(bb)
        write_out(BAS(intb), intb)







