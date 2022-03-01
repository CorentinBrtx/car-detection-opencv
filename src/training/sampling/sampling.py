from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .negative_sampling import generate_negative_samples
from .positive_sampling import generate_positive_samples


def generate_samples(
    annotation_file: str,
    size: Tuple[int, int] = (64, 64),
    neg_samples_per_image: int = 30,
    neg_samples_max_tries: int = 500,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate positive and negative samples from a given annotation file.

    Parameters
    ----------
    annotation_file : str
        Path to the annotation file, with a column "frame_id" containing the paths
        to the images, and a column "bounding_boxes" containing the bounding boxes
        for the corresponding image.
    size : Tuple[int, int], optional
        Size of the samples to generate, by default (64, 64)
    neg_samples_per_image : int, optional
        Number of negative samples to generate per image, by default 30
    neg_samples_max_tries : int, optional
        Maximum tries to generate negative samples for each image, by default 500

    Returns
    -------
    pos_samples, neg_samples : Tuple[List[np.ndarray], List[np.ndarray]]
        Tuple of lists of positive and negative samples.
    """
    pos_samples = []
    neg_samples = []

    annotations = pd.read_csv(annotation_file, converters={"frame_id": str, "bounding_boxes": str})

    for _, row in tqdm(annotations.iterrows(), total=len(annotations)):
        img = cv2.imread(row["frame_id"])
        bb = row["bounding_boxes"].split(" ")
        bounding_boxes = []
        for i in range(len(bb) // 4):
            bounding_boxes.append(
                [int(bb[i * 4]), int(bb[i * 4 + 1]), int(bb[i * 4 + 2]), int(bb[i * 4 + 3])]
            )

        pos_samples += generate_positive_samples(img, bounding_boxes, min_size=size)
        neg_samples += generate_negative_samples(
            img,
            size=size,
            n_samples=neg_samples_per_image,
            bounding_boxes=bounding_boxes,
            max_tries=neg_samples_max_tries,
        )

    return pos_samples, neg_samples
