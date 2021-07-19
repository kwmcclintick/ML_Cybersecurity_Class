from softmax_kwm import *
import numpy as np
import sys
from sklearn.datasets import make_classification
from sklearn import preprocessing

import scipy.io

'''
AUTHOR: KWM
COMMENT: performs a 2 layer softmax regression
'''



#-------------------------------------------------------------------------
def test_softmax_regression():
    # import and create training data set
    wk1 = scipy.io.loadmat('week1.mat')
    wk1 = wk1['week1']
    wk2 = scipy.io.loadmat('week2.mat')
    wk2 = wk2['week2']
    wk3 = scipy.io.loadmat('week3.mat')
    wk3 = wk3['week3']
    wk4 = scipy.io.loadmat('week4.mat')
    wk4 = wk4['week4']
    X_tr = np.concatenate((wk1[:,0:4], wk2[:,0:4], wk3[:,0:4], wk4[:,0:4]), axis=0)
    y_tr = np.concatenate((wk1[:,4], wk2[:,4], wk3[:,4], wk4[:,4]),axis=0)
    # set labels to be 0,1,...,C
    y_tr = y_tr - min(y_tr)
    # Make the data zero mean and unit variance such that the gradient is more spherical and converges faster
    X_tr = preprocessing.scale(X_tr)

    # import validation set
    wk5 = scipy.io.loadmat('week5.mat')
    wk5 = wk5['week5']
    X_te = wk5[:, 0:4]
    y_te = wk5[:, 4]

    # do the same to test data as training data
    X_te = preprocessing.scale(X_te)
    y_te = y_te - min(y_te)

    # train
    w,b = train(X_tr, y_tr, X_te, y_te, alpha=1e-1, n_epoch=50)
    np.save('w_trained.npy', w)
    np.save('b_trained.npy', b)
    # predict
    Y, P = predict(X_tr, w, b)
    accuracy = sum(Y == y_tr)/len(y_tr)
    print('Training accuracy:', accuracy)
    Y, P = predict(X_te, w, b)
    accuracy = sum(Y == y_te)/len(y_te)
    print('Test accuracy:', accuracy)

def predict_using_trained_weights():

    # import and create training data set
    wk1 = scipy.io.loadmat('week1.mat')
    wk1 = wk1['week1']
    wk2 = scipy.io.loadmat('week2.mat')
    wk2 = wk2['week2']
    wk3 = scipy.io.loadmat('week3.mat')
    wk3 = wk3['week3']
    wk4 = scipy.io.loadmat('week4.mat')
    wk4 = wk4['week4']
    X_tr = np.concatenate((wk1[:, 0:4], wk2[:, 0:4], wk3[:, 0:4], wk4[:, 0:4]), axis=0)
    y_tr = np.concatenate((wk1[:, 4], wk2[:, 4], wk3[:, 4], wk4[:, 4]), axis=0)
    # set labels to be 0,1,...,C
    y_tr = y_tr - min(y_tr)
    # Make the data zero mean and unit variance such that the gradient is more spherical and converges faster
    X_tr = preprocessing.scale(X_tr)

    # import validation set
    wk5 = scipy.io.loadmat('week5.mat')
    wk5 = wk5['week5']
    X_te = wk5[:, 0:4]
    y_te = wk5[:, 4]

    # do the same to test data as training data
    X_te = preprocessing.scale(X_te)
    y_te = y_te - min(y_te)

    # load trained weights
    w = np.load('w_trained.npy')
    b = np.load('b_trained.npy')

    # predict
    Y, P = predict(X_tr, w, b)
    accuracy = sum(Y == y_tr) / len(y_tr)
    print('Training accuracy:', accuracy)
    Y, P = predict(X_te, w, b)
    accuracy = sum(Y == y_te) / len(y_te)
    print('Test accuracy:', accuracy)