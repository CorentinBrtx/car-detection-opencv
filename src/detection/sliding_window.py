from typing import Tuple

import numpy as np


def sliding_window(image: np.ndarray, stepSize: int, windowSize: Tuple[int, int]):
    """Create a sliding window over an image."""
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield ([x, y, *windowSize], image[y : y + windowSize[1], x : x + windowSize[0]])
