import random
import numpy as np


"""
Transforms images with given indices
"""
def transform_images(images, indices):
    print indices
    print type(images)
    return images[:, indices]


"""
Function for calculating the error rate.
the argument pred and true_labels are matrix
"""
def calc_error(pred, true_label):
    unmatch = np.array(pred)!=np.array(true_label)
    return 1.0 * sum(unmatch) / len(pred)
