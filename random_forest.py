from decision_tree import build_decision_tree
from helpers import transform_images
import random
import numpy as np

def build_forest(images, labels, n_trees):
    forest = list() # list of trees
    for i in range(n_trees):
        feat_ind = sorted(random.sample(range(0,57),50))
        new_imgs = np.array(images)[:, feat_ind].tolist()
        #print new_img
        t = build_decision_tree(new_imgs, labels)
        forest.append((t, feat_ind))
        t.display()
    return forest
