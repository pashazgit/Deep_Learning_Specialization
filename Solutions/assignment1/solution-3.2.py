#!/usr/bin/env python3
# solution-3.2.py ---
#
# Filename: solution-3.2.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Wed Dec 20 10:46:31 2017 (-0800)
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
from PIL import Image

if __name__ == "__main__":

    img = np.array(Image.open("input.jpg"))
    img_inv = 255 - img

    with h5py.File("output.h5", "w") as ofp:
        ofp.create_group("data")
        ofp["data"]["image"] = img_inv

    plt.figure()
    plt.imshow(img)
    plt.figure()
    plt.imshow(img_inv)
    plt.show()

    # import IPython
    # IPython.embed()

#
# solution-3.2.py ends here
