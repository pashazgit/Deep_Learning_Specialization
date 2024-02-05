#!/usr/bin/env python3
# solution-3.4.py ---
#
# Filename: solution-3.4.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Wed Dec 20 11:52:06 2017 (-0800)
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


import h5py
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    with h5py.File("filtered.h5", "r") as ifp:
        filtered = np.asarray(ifp["data"]["image"])

    scores = np.sort(filtered.flatten())[::-1]
    idx_th = round(len(scores)*0.05)
    th = scores[idx_th]

    thresholded = filtered * (filtered >= th)

    plt.figure()
    plt.imshow(thresholded, cmap="gray")
    plt.show()

    # import IPython
    # IPython.embed()


#
# solution-3.4.py ends here
