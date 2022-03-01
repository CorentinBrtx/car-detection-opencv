from typing import List, Tuple

import numpy as np


def generate_positive_samples(
    img: np.ndarray, bounding_boxes: List[List[int]], min_size: Tuple[int, int]
) -> List[np.ndarray]:
    """
    Extract positive samples from a given image.

    Parameters
    ----------
    img : np.ndarray
        The image to extract positive samples from.
    bounding_boxes : List[List[int]]
        Bounding boxes to extract samples from.
    min_size : Tuple[int, int]
        Minimum size of the samples. Any bounding box smaller than this will be ignored.

    Returns
    -------
    positive_samples : List[np.ndarray]
        List of extracted positive samples.
    """
    samples = []

    for box in bounding_boxes:
        x, y, w, h = box
        if w >= min_size[0] and h >= min_size[1]:
            samples.append(img[y : y + h, x : x + w])

    return samples
