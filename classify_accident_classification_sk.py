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



metric_opts = ["acc", "bal"] #https://scikit-learn.org/stable/modules/model_evaluation.html
hyperopt_metric = "bal"
knn_opt_calls = 30
svm_opt_calls = 50
tree_opt_calls = 20

def loss(Ypred, Ypred_prob, Ytrue, metric = None): 
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
    def knn_loss(Xtrain, Xvalid, Ytrain, Yvalid, hopts):
        model = KNeighborsClassifier(**hopts)
        model.fit(Xtrain, Ytrain)
        Ypred = model.predict(Xvalid)
        Ypred_prob = model.predict_proba(Xvalid)

        return loss(Ypred, Ypred_prob, Yvalid)

    dim_n_neighbors = Integer(low = 1, high = int(50), name = "n_neighbors")
    dims = [dim_n_neighbors]

    @use_named_args(dimensions = dims)
    def fitness(n_neighbors):
        hopts = {"n_neighbors" : n_neighbors}
        return -knn_loss(Xtrain, Xvalid, Ytrain, Yvalid, hopts)

    initial_guess = [4]
    search_result = gp_minimize(func = fitness, dimensions = dims, n_calls = knn_opt_calls,
            x0 = initial_guess)

    hopts = {"n_neighbors" : search_result.x[0]}

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
    #for i in range(Ytest.size):
    #    print(iYtest[i].rjust(30), iYpred[i].rjust(30))

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
    def svm_loss(Xtrain, Xvalid, Ytrain, Yvalid, hopts):
        model = SVC(probability = True,**hopts)
        model.fit(Xtrain, Ytrain)
        Ypred = model.predict(Xvalid)
        Ypred_prob = model.predict_proba(Xvalid)

        return loss(Ypred, Ypred_prob, Yvalid)

    dim_C = Real(low = 0.1, high = 12, name = "C")
    dim_kernel = Categorical(["linear", "poly", "rbf", "sigmoid"], name = "kernel")
    dims = [dim_C, dim_kernel]

    @use_named_args(dimensions = dims)
    def fitness(C, kernel):
        hopts = {"C" : C, "kernel" : kernel}
        return -svm_loss(Xtrain, Xvalid, Ytrain, Yvalid, hopts)

    initial_guess = [1., "rbf"]
    search_result = gp_minimize(func = fitness, dimensions = dims, n_calls = svm_opt_calls,
            x0 = initial_guess)

    hopts = {"C" : search_result.x[0],
             "kernel" : search_result.x[1]}

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
    def tree_loss(Xtrain, Xvalid, Ytrain, Yvalid, hopts):
        model = DecisionTreeClassifier(**hopts)
        model.fit(Xtrain, Ytrain)
        Ypred = model.predict(Xvalid)
        Ypred_prob = model.predict_proba(Xvalid)

        return loss(Ypred, Ypred_prob, Yvalid)

    dim_criterion = Categorical(["gini", "entropy"], name = "criterion")
    dim_splitter = Categorical(["best", "random"], name = "splitter")
    dim_max_depth = Integer(low = 1, high = 30, name = "max_depth")
    dims = [dim_criterion, dim_splitter, dim_max_depth]

    @use_named_args(dimensions = dims)
    def fitness(criterion, splitter, max_depth):
        hopts = {"criterion" : criterion, "splitter" : splitter, "max_depth" : max_depth}
        return -tree_loss(Xtrain, Xvalid, Ytrain, Yvalid, hopts)

    initial_guess = ["gini", "best", 2]
    search_result = gp_minimize(func = fitness, dimensions = dims, n_calls = tree_opt_calls,
            x0 = initial_guess)

    hopts = {"criterion" : search_result.x[0],
             "splitter" : search_result.x[1],
             "max_depth" : search_result.x[2]}

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

    return train_losses, test_losses

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
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(rY)

    # train, test, verification split
    Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest  = split(X, Y)
    dsets = [Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest]

    rs = [knn(*dsets, scaler, label_encoder),
    logreg(*dsets, scaler, label_encoder),
    lda(*dsets, scaler, label_encoder),
    svm(*dsets, scaler, label_encoder),
    tree(*dsets, scaler, label_encoder),
    gpc(*dsets, scaler, label_encoder)]

    names = ["knn", "log regr", "lda", "svm", "tree", "gpc"]


    cols = ["Method"] + ["train_" + a for a in metric_opts] \
            + ["test_" + a for a in metric_opts]
    cols = [a.ljust(15) for a in cols]
    print("".join(cols))
    for i, r in enumerate(rs):
        pt = ""
        pt += names[i].ljust(15)
        for m in metric_opts:
            pt += ("%.4f"%r[0][m]).ljust(15)
        for m in metric_opts:
            pt += ("%.4f"%r[1][m]).ljust(15)
        print(pt)



    print("Rutime:", time.time() - start, "s")

