from decision_tree import build_decision_tree
import math
import numpy

def final_hyp(x, X, Y, it):
    trees = adaboost(X, Y, it)
    s = 0
    for t in trees:
        s += t.alpha * t.classify(t.root, x)
    return 0 if s < 0 else 1


def adaboost(X, Y, iterations=10):
    Y = Y.astype(numpy.int16, copy=False)
	d = [1.0 / len(Y)] * len(Y)
    trees = []
	for i in range(iterations):
        x_sampled, y_sampled = sample_distr(d, X, Y)
		t = build_decision_tree(x_sampled, y_sampled, d=2):
		error = calc_error(t)
		alpha = 0.5 * math.log((1.0 - err) / err)
        t.set_alpha(alpha)
        trees += [t]
        for i in range(d):
            d[i] = d[i] * exp(- alpha * y_sampled[i] * 
                t.classify(t.root, x_sampled[i]))
        d = normalize(d)
    return trees


def sample_distr(distr, X, Y):
    XY = np.append(X, Y, axis=1)
    


def calc_error(tree):
	pred = t.classify(X) # classify the features
    pred[pred==0] = -1
    indices = pred != Y # incorrect predicts
    err = sum(d[indices])


def update(d, t, error):
   err = t.classify(X)
	alpha = 1/2 math.log((1.0-err)/err)
   return 


def normalize(d):
    pass
