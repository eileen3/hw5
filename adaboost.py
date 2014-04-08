from decision_tree import build_decision_tree
import math


def adaboost(X, Y, iterations=10):
	d = [1.0 / len(Y)] * len(Y)
	for i in range(iterations):
		t = build_decision_tree(X, Y, d):
		error = calc_error(t)
        pred = t.classify(X) # classify the features
        pred[pred==0] = -1
        Y[Y==0] = -1 # switch the labels from 0 to 1
        indices = pred != Y # incorrect predicts
        err = sum(d[indices])
		alpha = 0.5 * math.log((1.0 - err) / err)
        d = d * exp(- alpha * Y * pred) # pointwise multiplication
        d = normalize(d)
	return alpha

"""
strong hypothesis. takes alpha a, feature X, and weak hypothesis h
"""
def strong_hyp(a, X, h):
    
    return sign

#def calc_error(tree):
#	
#
#def update(d, t, error):
#    err = t.classify(X)
#	alpha = 1/2 math.log((1.0-err)/err)
#    return 

