# logistic_regression.py ---
#
# Filename: logistic_regression.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Mon Jan 15 13:07:21 2018 (-0800)
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
# Copyright (C), Visual Computing Group @ University of Victoria.

# Code:

import numpy as np


def model_loss(W, b, x, y):
    """Loss function.

    Parameters
    ----------
    W : ndarray
        The weight parameters of the linear classifier. D x C, where C is the
        number of classes, and D is the dimenstion of input data.

    b : ndarray
        The bias parameters of the linear classifier. C, where C is the number
        of classes.

    x : ndarray
        Input data that we want to predic the labels of. NxD, where D is the
        dimension of the input data.

    y : ndarray
        Ground truth labels associated with each sample. N numbers where each
        number corresponds to a class.

    Returns
    -------
    loss : float
        The average loss coming from this model. In the lecture slides,
        represented as \frac{1}{N}\sum_i L_i.

    """

    # Scores for all class (N, 10)
    s_all = np.matmul(x, W) + b
    # Do exponential and normalize to get probs
    probs = np.exp(s_all - s_all.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    # For the cross entropy case, we will return the probs in loss_c

    # Compute loss
    loss = np.mean(-np.log(probs[np.arange(len(y)), y]))

    return loss


#
# logistic_regression.py ends here
