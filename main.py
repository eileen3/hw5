import scipy.io
from decision_tree import build_decision_tree
from adaboost import adaboost, final_hyp
import random_forest as rf
from sklearn.cross_validation import KFold
from sklearn import cross_validation
import helpers
import numpy as np

def dt_crossvalidate(X, Y):
    x_train, x_test, y_train, y_test = \
        cross_validation.train_test_split(X, Y, test_size=0.1, random_state=0)
    t = build_decision_tree(x_train, y_train, d=1)
    return t.score(x_test, y_test)

def random_forest_crossvalidate():
    ### Code for cross-validation of random forest
    kf = KFold(len(mat['ytrain']), n_folds=5, indices=False)
    for train, test in kf:
        f = rf.build_forest(mat['Xtrain'][train], mat['ytrain'][train],10) 
        pred = rf.classify_forest(mat['Xtrain'][test], f)
        print helpers.calc_error(pred, mat['ytrain'][test])

def adaboost_crossvalidate(X, Y):
    x_train, x_test, y_train, y_test = \
        cross_validation.train_test_split(X, Y, test_size=0.1, random_state=0)
    trees = adaboost(x_train, y_train, 100)
    correct = 0
    for i in range(len(x_test)):
        if final_hyp(x_test[i], trees) == y_test[i][0]:
            correct += 1
    return float(correct) / len(x_test)

def random_forest_predict_the_test_set():
    ### Now creates a random forest in general
    f = rf.build_forest(mat['Xtrain'],mat['ytrain'],75)
    pred = rf.classify_forest(mat['Xtest'], f)
    np.savetxt('forest.out', pred, fmt='%d')

if __name__ == '__main__':
    mat = scipy.io.loadmat('spam.mat')
    X = mat['Xtrain']
    Y = mat['ytrain']
    #print dt_crossvalidate(mat['Xtrain'], mat['ytrain'])
    #t = build_decision_tree(mat['Xtrain'], mat['ytrain'])
    #t.display()
    #random_forest_predict_the_test_set()
    print adaboost_crossvalidate(X, Y)
