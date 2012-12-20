from __future__ import division

from skimage import io
from skimage import transform
from skimage import img_as_float
from skimage.exposure import rescale_intensity

import matplotlib.pyplot as plt
import numpy as np
import os

phantom = img_as_float(io.imread('phantom.png'))
atlas = img_as_float(io.imread('atlas.png')[..., :3])


def choose_corresponding_points(img0, img1):
    """Utility function for finding corresponding features in images.

    Alternately click on image 0 and 1, indicating the same feature.

    """
    f, (ax0, ax1) = plt.subplots(1, 2)
    ax0.imshow(img0)
    ax1.imshow(img1)

    coords = plt.ginput(16, timeout=0)

    np.savez('_reg_coords.npz', source=coords[::2], target=coords[1::2])

    plt.close()


# Re use previous coordinates, if found
if not os.path.exists('_reg_coords.npz'):
    choose_corresponding_points(phantom, atlas)


coords = np.load('_reg_coords.npz')

tf = transform.estimate_transform('polynomial',
                                  coords['source'], coords['target'],
                                  order=2)

atlas_warped = transform.warp(atlas, inverse_map=tf,
                              output_shape=phantom.shape)

f, (ax0, ax1, ax2) = plt.subplots(1, 3,
                                  subplot_kw={'xticks': [], 'yticks': []})
ax0.imshow(phantom)
x, y = coords['source'].T
ax0.plot(x, y, 'b.')

ax1.imshow(atlas)
x, y = coords['target'].T
ax1.plot(x, y, 'b.')

ax2.imshow(rescale_intensity(phantom * atlas_warped, in_range=(0, 0.5)))

plt.show()
