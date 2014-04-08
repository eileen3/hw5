import scipy.io
from decision_tree import build_decision_tree
from sklearn import cross_validation
#from random_forest import build_forest
import numpy as np

def cross_validate(X, Y):
	x_train, x_test, y_train, y_test = \
		cross_validation.train_test_split(X, Y, test_size=0.1, random_state=0)
	t = build_decision_tree(x_train, y_train)
	return t.score(x_test, y_test)


if __name__ == '__main__':
	mat = scipy.io.loadmat('spam.mat')
	#print cross_validate(mat['Xtrain'], mat['ytrain'])
	t = build_decision_tree(mat['Xtrain'], mat['ytrain'])
	t.display()
    #f = build_forest(mat['Xtrain'], mat['ytrain'],100) 
