import os
import pickle

import cv2
import pandas as pd
from tqdm import tqdm

from ..utils import bounding_boxes_to_mask, run_length_encoding
from .detect import detect_cars


def make_submission_file(
    test_folder: str,
    model_path: str,
    target_path: str = "submission.csv",
    H: int = 720,
    W: int = 1280,
    **kwargs
):
    """
    Create a submission file for the Kaggle Challenge for the images in the test_folder.

    Parameters
    ----------
    test_folder : str
        folder containing the images to detect cars in.
    model_path : str
        path to the trained model.
    target_path : str, optional
        path to save the submission file to, by default "submission.csv"
    H : int, optional
        height of the images, by default 720
    W : int, optional
        width of the images, by default 1280
    **kwargs : additional keyword arguments to pass to the detect_cars function (see detection/detect.py for details)
    """

    test_files = sorted(os.listdir(test_folder))

    rows = []

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    for file_name in tqdm(test_files):

        img = cv2.imread(os.path.join(test_folder, file_name))
        bounding_boxes = detect_cars(img, model, **kwargs)

        if len(bounding_boxes) == 0:
            rle = ""
        else:
            rle = run_length_encoding(bounding_boxes_to_mask(bounding_boxes, H, W))
        rows.append([os.path.join(test_folder, file_name), rle])

    df_prediction = pd.DataFrame(columns=["Id", "Predicted"], data=rows).set_index("Id")
    df_prediction.to_csv(target_path)
