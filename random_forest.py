from decision_tree import build_decision_tree
from helpers import transform_images
import helpers
import random
import numpy as np
from scipy import stats

"""
build_forest() builds random forest
with randomly chosen 3/4 of the input
data points and 35 out of 57 features
"""
def build_forest(images, labels, n_trees):
    forest = list() # list of trees
    num_data_points = images.shape[0]
    for i in range(n_trees):
        feat_ind = sorted(random.sample(range(0,57),35)) # selected feature
        sample_ind = random.sample(range(num_data_points),num_data_points * 3 / 4) # sample indexes
        new_labels = labels[sample_ind]
        new_imgs = images[sample_ind, :]
        new_imgs = new_imgs[:, feat_ind]
        
        t = build_decision_tree(new_imgs, new_labels)
        forest.append((t, feat_ind))
    return forest # list of trees and features it used

"""
Given a images and a random forest,
it classifies the input images
and returns a vector matrix
"""
def classify_forest(images, forest):
    pred_mat = list()
    for tree, feat_ind in forest:
        pred = list()
        for img in images:
            pred.append(tree.classify(tree.root,img[feat_ind]))
        pred_mat.append(pred)
    pred = stats.mode(np.array(pred_mat))
    return np.int_(pred[0][0].reshape(pred[0][0].shape[0],1))
