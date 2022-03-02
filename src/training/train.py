import argparse
import logging
from typing import Tuple

import cv2
from sklearn.base import ClassifierMixin
from src.hog import compute_features
from src.training.sampling.sampling import generate_samples
from src.training.svm import train_and_save_model


def train(
    annotation_file: str, model_path: str, samples_size: Tuple[int, int] = (64, 64), **kwargs
) -> ClassifierMixin:
    """
    Train an SVC model and save it to the specified path.
    The model is trained using positive and negative samples extracted
    from the images referenced in the annotation file.

    Parameters
    ----------
    annotation_file : str
        Path to the annotation file.
    model_path : str
        Path to save the model to.
    samples_size : Tuple[int, int], optional
        Size of the samples to extract from the images, by default (64, 64)

    Returns
    -------
    model : ClassifierMixin
        Trained model.
    """
    logging.info("Generating training samples...")

    pos_samples, neg_samples = generate_samples(annotation_file, size=samples_size, **kwargs)

    logging.info(
        "Generated %i positive samples and %i negative samples", len(pos_samples), len(neg_samples)
    )

    hog = cv2.HOGDescriptor(
        _winSize=(64, 64), _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9
    )

    logging.info("Computing features...")

    pos_samples_features = compute_features(hog, pos_samples)
    neg_samples_features = compute_features(hog, neg_samples)

    logging.info("Training model...")

    model = train_and_save_model(pos_samples_features, neg_samples_features, model_path)

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a new model for car detection")
    parser.add_argument("annotation_file", type=str, help="Path to the annotation file")
    parser.add_argument("model_path", type=str, help="Path to save the model to")

    args = parser.parse_args()

    train(args.annotation_file, args.model_path)
