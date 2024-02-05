#!/usr/bin/env python3
# solution-3.3.py ---
#
# Filename: solution-3.3.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Wed Dec 20 10:57:41 2017 (-0800)
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

import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter

if __name__ == "__main__":

    with h5py.File("output.h5", "r") as ifp:
        img_inv = np.asarray(ifp["data"]["image"])

    img_inv = np.mean(img_inv, axis=-1)

    k1 = int(sys.argv[1])
    k2 = int(sys.argv[2])

    res1 = gaussian_filter(img_inv, k1)
    res2 = gaussian_filter(img_inv, k2)

    res = res2 - res1

    res -= res.min()
    res /= res.max()

    plt.figure()
    plt.imshow(res, cmap="gray")

    with h5py.File("filtered.h5", "w") as ofp:
        ofp.create_group("data")
        ofp["data"]["image"] = res

    plt.show()
    # import IPython
    # IPython.embed()


#
# solution-3.3.py ends here
