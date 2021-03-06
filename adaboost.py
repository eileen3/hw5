from decision_tree import build_decision_tree
from scipy import stats
import math
import numpy as np
from numpy.random import random_sample
from sklearn.preprocessing import normalize

def final_hyp(x, trees):
    """
    The final hypothesis. Classifies x based on trees generated
    by adaboost method.
    """
    s = 0
    for t in trees:
        s += t.alpha * t.classify(t.root, x)
    return 0 if s < 0 else 1


def adaboost(X, Y, iterations=10):
    """Performs adaboost."""
    d = [1.0 / len(Y)] * len(Y)
    trees = []
    Y = Y.astype(np.int16)
    Y[Y==0] = -1
    minimum_weight = float('inf')
    for i in range(iterations):
        x_sampled, y_sampled = sample_distr(d, X, Y)
        t = build_decision_tree(x_sampled, y_sampled, d=0)
        err = calc_error(t, d, x_sampled, y_sampled)
        if err == 0:
            alpha = minimum_weight / 100
        else:
            alpha = 0.5 * math.log((1.0 - err) / err)
            if alpha < minimum_weight:
                minimum_weight = alpha
        t.set_alpha(alpha)
        trees += [t]
        for i in range(len(d)):
            d[i] = d[i] * math.exp(- alpha * y_sampled[i] * 
                t.classify(t.root, x_sampled[i]))
        total = sum(d)
        d = [float(d[i]) / total for i in range(len(d))]
    return trees


def sample_distr(distr, X, Y):
    """Resamples data based on distribution given."""
    XY = np.append(X, Y, axis=1)
    bins = np.add.accumulate(distr)
    new_XY = XY[np.digitize(random_sample(len(X)), bins)]
    x, y = np.hsplit(new_XY, [57])
    return (x, y)


def calc_error(tree, d, x, y):
    """Calculates the error of weak hypothesis."""
    error = 0
    y = y.reshape(-1).tolist()
    for i in range(len(x)):
        pred = tree.classify(tree.root, x[i]) # classify the features
        if int(pred) != int(y[i]):
            error += d[i]
    return error
