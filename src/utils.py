from typing import List

import numpy as np


def run_length_encoding(mask: np.ndarray):
    """
    Produces run length encoding for a given binary mask
    """

    # find mask non-zeros in flattened representation
    non_zeros = np.nonzero(mask.flatten())[0]
    padded = np.pad(non_zeros, pad_width=1, mode="edge")

    # find start and end points of non-zeros runs
    limits = (padded[1:] - padded[:-1]) != 1
    starts = non_zeros[limits[:-1]]
    ends = non_zeros[limits[1:]]
    lengths = ends - starts + 1

    return " ".join([f"{s} {l}" for s, l in zip(starts, lengths)])


def bounding_boxes_to_mask(bounding_boxes: List[List[int]], H: int, W: int) -> np.ndarray:
    """
    Convert bounding boxes to a binary mask.

    Parameters
    ----------
    bounding_boxes : List[List[int]]
        List of bounding boxes, in the format [x, y, w, h].
    H : int
        Height of the image.
    W : int
        Width of the image.

    Returns
    -------
    mask : np.ndarray
        Binary mask.
    """
    mask = np.zeros((H, W))

    for x, y, dx, dy in bounding_boxes:
        mask[y : y + dy, x : x + dx] = 1

    return mask
