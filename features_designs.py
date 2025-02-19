import numpy as np

def feature_affine_design(X):
    return np.column_stack((np.ones(X.shape[0]), X))
