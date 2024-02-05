# linear_svm.py ---
#
# Filename: linear_svm.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Sun Jan 14 20:45:06 2018 (-0800)
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
    # Score for the correct class (N, )
    s_y = s_all[np.arange(len(y)), y]
    # Make Nx1 to sub from s_all
    s_y = np.reshape(s_y, (-1, 1))
    # Loss per class (including the correct one)
    loss_c = np.maximum(0, s_all - s_y + 1)
    # Compute loss by averaging of samples, summing over classes. We subtract 1
    # after the sum, since the correct label always returns 1 in terms of the
    # per-class loss, and should be excluded from the final loss
    loss = np.mean(np.sum(loss_c, axis=1) - 1, axis=0)

    return loss


#
# linear_svm.py ends here
