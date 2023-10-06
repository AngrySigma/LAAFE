
import numpy as np
import pandas as pd


def add_new_features(dataset):
    # Get the sepal area (sepal length * sepal width)
    sepal_area = dataset[:, 0] * dataset[:, 1]
    
    # Get the petal area (petal length * petal width)
    petal_area = dataset[:, 2] * dataset[:, 3]
    
    # Get the ratio of petal length to petal width
    petal_ratio = dataset[:, 2] / dataset[:, 3]
    
    # Get the ratio of sepal length to sepal width
    sepal_ratio = dataset[:, 0] / dataset[:, 1]

    return sepal_area, petal_area, petal_ratio, sepal_ratio
    # Concatenate the new features 