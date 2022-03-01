from typing import List

import cv2
import numpy as np


def compute_features(hog_descriptor: cv2.HOGDescriptor, imgs: List[np.ndarray]) -> List[np.ndarray]:
    """
    Compute HOG features for a list of images.

    Parameters
    ----------
    hog_descriptor : cv2.HOGDescriptor
        The HOG descriptor to use for computing features.
    imgs : List[np.ndarray]
        List of images to compute features for.

    Returns
    -------
    features : List[np.ndarray]
        List of HOG features vector for each image.
    """
    features = []
    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 64))
        f = hog_descriptor.compute(img)
        features.append(f)
    return features
