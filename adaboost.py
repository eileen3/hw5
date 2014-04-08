from decision_tree import build_decision_tree
import math


def adaboost(X, Y, iterations=10):
	d = [1.0 / len(Y)] * len(Y)
	for i in range(iterations):
		t = build_decision_tree(X, Y, d):
		error = calc_error(t)
		alpha = 0.5 * math.log((1 - error) / error)
		d = update(d, t, error)
	return t

def calc_error(tree):
	

def update(d, t, error):
	pass

