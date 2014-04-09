import scipy.io
from decision_tree import build_decision_tree
import random_forest as rf 
from sklearn.cross_validation import KFold
import helpers
import numpy as np

def cross_validate(X, Y):
	x_train, x_test, y_train, y_test = \
		cross_validation.train_test_split(X, Y, test_size=0.1, random_state=0)
	t = build_decision_tree(x_train, y_train)
	return t.score(x_test, y_test)

def random_forest_crossvalidate():
    ### Code for cross-validation of random forest
    kf = KFold(len(mat['ytrain']), n_folds=5, indices=False)
    for train, test in kf:
        f = rf.build_forest(mat['Xtrain'][train], mat['ytrain'][train],10) 
        pred = rf.classify_forest(mat['Xtrain'][test], f)
        print helpers.calc_error(pred, mat['ytrain'][test])
    
def random_forest_predict_the_test_set():
    ### Now creates a random forest in general
    f = rf.build_forest(mat['Xtrain'],mat['ytrain'],75)
    pred = rf.classify_forest(mat['Xtest'], f)
    np.savetxt('forest.out', pred, fmt='%d')

if __name__ == '__main__':
    mat = scipy.io.loadmat('spam.mat')
    #print cross_validate(mat['Xtrain'], mat['ytrain'])
    #t = build_decision_tree(mat['Xtrain'], mat['ytrain'])
    # t.display()
    random_forest_predict_the_test_set()
