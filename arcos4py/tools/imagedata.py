"""Toolset to convert image data to a dataframe for use with ARCOS."""

import pandas as pd
import numpy as np
from typing import Union, Tuple
from scipy.ndimage import gaussian_filter, median_filter



def image_to_dataframe(image: np.ndarray) -> pd.DataFrame:
    """Converts a 2d image series to a dataframe with columns for x, y, and intensity.
    to be used with ARCOS.

    Arguments:
        image (np.ndarray): Image to convert.

    Returns (pd.DataFrame): Dataframe with image data.
    """
    df_all = []
    for idx, i in enumerate(image):
        indicesArray = np.moveaxis(np.indices(i.shape), 0, 2)
        allArray = np.dstack((indicesArray, i)).reshape((-1, 3))
        df = pd.DataFrame(allArray, columns=["y", "x", "value"])
        df_all.append(df)
        df['t'] = idx        
    df_all = pd.concat(df_all)
    df_all['track_id'] = np.tile(np.arange(len(df_all[df_all.t==0])), len(df_all.t.unique()))
    return df_all

def remove_image_background(image: np.ndarray, filter_type: str = 'gaussian', size: Union[int, Tuple] = 10  ) -> np.ndarray:
    """Removes background from d2 images. Assumes the two last axes are x and y.

    Arguments:
        image (np.ndarray): Image to remove background from.
        filter_type (Union[str, function]): Filter to use to remove background. Can be one of ['median', 'gaussian'].
        filter_size (int, Tuple): Size of filter to use. For median filter, this is the size of the window. 
            For gaussian filter, this is the standard deviation. If a tuple is passed then the first value is used for t, the second for x, and the third for y.

    Returns (np.ndarray): Image
    """
    ## correct images with a gaussian filter applied over time wiht a sigma shift
    if filter_type not in ['median', 'gaussian'] and not callable(filter_type):
        raise ValueError('Filter type must be one of ["median", "gaussian"]')
    orig_image = image.copy()
    shift = 20
    s_2 = shift // 2
    if filter_type  in ['median', 'gaussian']:
        if isinstance(size, int):
            size = (size, 1, 1)
        if len(size) == 2:
            size = (size[0], size[1], size[1])
        if len(size) == 3:
            size = (size[0], size[1], size[2])

    if filter_type == 'median':
        filtered = median_filter(orig_image, size=size)
    elif filter_type == 'gaussian':
        filtered = gaussian_filter(orig_image, sigma=size)

    corr = orig_image - filtered
    corr = corr[s_2:-s_2]
    return corr

def remove_edge_artefacts(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Removes edge artefacts from an image.

    Arguments:
        image (np.ndarray): Image to remove artefacts from.
        mask (np.ndarray): Mask to use to remove artefacts.

    Returns (np.ndarray): Image with artefacts removed.
    """
    image = image.copy()
    image[~mask] = 0
    return image