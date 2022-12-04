"""Toolset to convert image data to a dataframe for use with ARCOS."""

from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, median_filter
from skimage.util import view_as_blocks
from scipy.ndimage import binary_erosion, binary_fill_holes


def _2d_image_to_dataframe(image: np.ndarray, thresh: int = 0) -> pd.DataFrame:
    """Converts a 2d image series to a dataframe with columns for x, y, and intensity.
    to be used with ARCOS.

    Arguments:
        image (np.ndarray): Image to convert. Assumes first axis is t and the two last axes are y and x.
        thresh (int): Threshold to use to remove background. Default is 0.

    Returns (pd.DataFrame): Dataframe with image data.
    """
    df_all = []
    for idx, i in enumerate(image):
        indicesArray = np.moveaxis(np.indices(i.shape), 0, 2)
        allArray = np.dstack((indicesArray, i)).reshape((-1, 3))
        df = pd.DataFrame(allArray, columns=["y", "x", "value"])
        df_all.append(df)
        df['t'] = idx
    df_all_combined = pd.concat(df_all)
    df_all_combined['track_id'] = np.tile(
        np.arange(len(df_all_combined[df_all_combined.t == 0])), len(df_all_combined.t.unique())
    )
    df_all_combined = df_all_combined[df_all_combined['value'] > thresh]
    return df_all_combined


def _3d_image_to_dataframe(image: np.ndarray, thresh: int = 0) -> pd.DataFrame:
    """Converts a 3d image series to a dataframe with columns for x, y, z, and intensity.
    to be used with ARCOS.

    Arguments:
        image (np.ndarray): Image to convert. Assumes first axis is t and the three last axes are z, y, and x.
        thresh (int): Threshold to use to remove background. Default is 0.

    Returns (pd.DataFrame): Dataframe with image data.
    """
    df_all = []
    for idx, i in enumerate(image):
        indicesArray = np.moveaxis(np.indices(i.shape), 0, 3)
        allArray = np.dstack((indicesArray, i)).reshape((-1, 4))
        df = pd.DataFrame(allArray, columns=["z", "y", "x", "value"])
        df_all.append(df)
        df['t'] = idx
    df_all_combined = pd.concat(df_all)
    df_all_combined['track_id'] = np.tile(
        np.arange(len(df_all_combined[df_all_combined.t == 0])), len(df_all_combined.t.unique())
    )
    df_all_combined = df_all_combined[df_all_combined['value'] > thresh]
    return df_all_combined


def image_to_dataframe(image: np.ndarray, thresh: int = 0):
    """Converts an image series to a dataframe with columns for x, y, z, and intensity. Assumes axis order (t, y, x) for 2d images and (t, z, y, x) for 3d images.


    Arguments:
        image (np.ndarray): Image to convert. Assumes first axis is t and the three last axes are z, y, and x.
        thresh (int): Threshold to use to remove background. Default is 0.

    Returns (pd.DataFrame): Dataframe with image data.
    """
    if image.ndim == 3:
        return _2d_image_to_dataframe(image, thresh)
    elif image.ndim == 4:
        return _3d_image_to_dataframe(image, thresh)
    else:
        raise ValueError(f'Image must have 3 or 4 dimensions. Image has {image.ndim} dimensions.')


def remove_image_background(
    image: np.ndarray, filter_type: str = 'gaussian', size: Union[int, Tuple] = (10, 1, 1)
) -> np.ndarray:
    """Removes background from images. Assumes axis order (t, y, x) for 2d images and (t, z, y, x) for 3d images.

    Arguments:
        image (np.ndarray): Image to remove background from.
        filter_type (Union[str, function]): Filter to use to remove background. Can be one of ['median', 'gaussian'].
        size (int, Tuple): Size of filter to use. For median filter, this is the size of the window.
            For gaussian filter, this is the standard deviation. If a single int is passed in, it is assumed to be the same for all dimensions.
            If a tuple is passed in, it is assumed to correspond to the size of the filter in each dimension.
            Default is (10, 1, 1).

    Returns (np.ndarray): Image with background removed.
        Along the first axis (t) half of the filter size is removed from the beginning and end respectively.
    """
    ## correct images with a gaussian filter applied over time wiht a sigma shift
    allowed_filters = ['median', 'gaussian']
    if filter_type not in allowed_filters:
        raise ValueError(f'Filter type must be one of {allowed_filters}.')
    orig_image = image.copy()
    shift = 20
    s_2 = shift // 2

    if isinstance(size, int):
        size = (size,) * image.ndim
    elif isinstance(size, tuple):
        if len(size) != image.ndim:
            raise ValueError(f'Filter size must have {image.ndim} dimensions.')
    else:
        raise ValueError(f'Filter size must be an int or tuple.')

    if filter_type == 'median':
        filtered = median_filter(orig_image, size=size)
    elif filter_type == 'gaussian':
        filtered = gaussian_filter(orig_image, sigma=size)

    corr = np.subtract(orig_image, filtered, dtype=np.float32)
    corr = corr[s_2:-s_2]
    return corr


def blockwise_median(a, blockshape):
    """Calculates the blockwise median of an array.

    Arguments:
        a (np.ndarray): Array to calculate blockwise median of.
        blockshape (Tuple): Shape of blocks to use.

    Returns (np.ndarray): Blockwise median of array.
    """
    assert a.ndim == len(blockshape), "blocks must have same dimensionality as the input image"
    assert not (np.array(a.shape) % blockshape).any(), "blockshape must divide cleanly into the input image shape"

    block_view = view_as_blocks(a, blockshape)
    assert block_view.shape[a.ndim :] == blockshape
    block_axes = [*range(a.ndim, 2 * a.ndim)]
    return np.median(block_view, axis=block_axes)


def mask_image(image: np.ndarray, mask: np.ndarray, mask_erosion_iterations: int = 1) -> np.ndarray:
    """Masks an image with a binary or label image. Can be used to
    remove edge artifacts that occur when using temporal filtering as
    implemented in remove_image_background.

    Arguments:
        image (np.ndarray): Image to mask.
        mask (np.ndarray): Mask to use.

    Returns (np.ndarray): Masked image.
    """
    if mask.shape != image.shape:
        raise ValueError(f'Mask must have same shape as image. Mask shape: {mask.shape}, Image shape: {image.shape}')
    if mask_erosion_iterations > 0:
        mask = binary_erosion(mask, iterations=mask_erosion_iterations)

    return np.where(mask > 0, image, 0)
