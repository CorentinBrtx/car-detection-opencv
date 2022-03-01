from typing import List, Tuple

import cv2
import numpy as np
from sklearn.base import ClassifierMixin

from ..hog import compute_features
from .sliding_window import sliding_window


def detect_cars(
    img: np.ndarray,
    model: ClassifierMixin,
    min_size: Tuple[int, int] = (64, 64),
    max_size: Tuple[int, int] = (512, 512),
) -> List[List[int]]:
    """
    Detect cars in an image using a trained model and a sliding window.

    Parameters
    ----------
    img : np.ndarray
        The image to detect cars in.
    model : ClassifierMixin
        The trained model to use for detection.
    min_size : Tuple[int, int], optional
        Minimum detection window size, by default (64, 64)
    max_size : Tuple[int, int], optional
        Maximum detection window size, by default (512, 512)

    Returns
    -------
    bounding_boxes : List[List[int]]
        Bounding boxes of detected cars, in the form of [x, y, width, height].
    """

    hog = cv2.HOGDescriptor(
        _winSize=(64, 64), _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9
    )

    bounding_boxes = []

    patches = []
    boxes = []

    for scale in range(1, max_size[0] // min_size[0]):
        for box, patch in sliding_window(
            img, max(16, min_size[0] * scale // 5), (min_size[0] * scale, min_size[1] * scale)
        ):
            boxes.append(box)
            patches.append(patch)

    features = compute_features(hog, patches)

    classification = model.predict(features)

    bounding_boxes = [boxes[i] for i in range(len(classification)) if classification[i] == 1]

    return bounding_boxes
