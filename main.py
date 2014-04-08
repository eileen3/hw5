import scipy.io
import math
import decision_tree
import adaboost

if __name__ == '__main__':
	mat = scipy.io.loadmat('spam.mat')
	t = build_decision_tree(mat['Xtrain'], mat['ytrain'])
