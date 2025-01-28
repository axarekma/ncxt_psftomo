import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def getslice(data, dim, slicenr=None):
    """Get slice of a n-dimensional array

    Arguments:
        data {ndarray} -- Input array.
        dim {int} -- dimension of the slicing

    Keyword Arguments:
        slicenr {int} -- Slice number, if not assigned gets the middle

    Returns:
        [ndarray] -- Output array
    """

    if slicenr is None:
        slicenr = int(data.shape[dim] / 2)
    assert -1 < slicenr < data.shape[dim], f"Index {slicenr} is out of range"

    return np.take(data, slicenr, axis=dim)


def volshow(data, slices=(None, None, None)):
    """Show preview of volume

    Arguments:
        data {ndarray} -- Input array.

    Keyword Arguments:
        slices {tuple} -- Shown slices, default is middle (default: {(-1, -1, -1)})
    """

    ndim = data.ndim
    assert ndim == 3, "Volume must be a 3d ndarray"

    _ = plt.figure(figsize=(13, 5))
    axes = [plt.subplot(gsi) for gsi in gridspec.GridSpec(1, ndim)]
    images = [getslice(data, d, s) for d, s in zip(range(ndim), slices)]

    for axis, image in zip(axes, images):
        axis.imshow(image)

