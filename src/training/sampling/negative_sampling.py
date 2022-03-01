from typing import List, Tuple

import numpy as np


def generate_negative_samples(
    img: np.ndarray,
    size: Tuple[int, int] = (64, 64),
    n_samples: int = 10,
    bounding_boxes: List[List[int]] = None,
    max_tries: int = 500,
) -> List[np.ndarray]:
    """
    Generate negative samples from a given image by sampling random patches
    while avoiding the specified bounding boxes.

    Parameters
    ----------
    img : np.ndarray
        The image to generate negative samples from.
    size : Tuple[int, int], optional
        Size of the samples, by default (64, 64)
    n_samples : int, optional
        Number of samples to generate, by default 10
    bounding_boxes : List[List[int]], optional
        Bounding boxes to avoid, by default None
    max_tries : int, optional
       Maximum tries to generate samples
       (avoid infinite loop if bounding_boxes take too much space), by default 500

    Returns
    -------
    negative_samples : List[np.ndarray]
        List of generated negative samples.
    """
    if bounding_boxes is None:
        bounding_boxes = []

    mask = np.zeros(img.shape[:2], np.uint8)
    mask_with_margin = np.zeros(img.shape[:2], np.uint8)

    for box in bounding_boxes:
        x, y, w, h = box
        mask[y : y + h, x : x + w] = 1
        mask_with_margin[max(0, y - size[0]) : y + h, max(0, x - size[1]) : x + w] = 1

    mask_with_margin[:, -size[1] :] = 1
    mask_with_margin[-size[0] :, :] = 1

    negative_samples = []
    tries = 0

    indices = np.transpose(np.nonzero(mask_with_margin == 0))

    while len(negative_samples) < n_samples and tries < max_tries:
        tries += 1

        top_left = indices[np.random.randint(0, len(indices))]

        if (
            mask[top_left[0] : top_left[0] + size[0], top_left[1] : top_left[1] + size[1]].sum()
            == 0
        ):
            negative_samples.append(
                img[top_left[0] : top_left[0] + size[0], top_left[1] : top_left[1] + size[1]]
            )

    return negative_samples
