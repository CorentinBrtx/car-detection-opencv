import pickle
from typing import List

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_and_save_model(
    pos_samples_features: List[np.ndarray], neg_samples_features: List[np.ndarray], model_path: str
) -> ClassifierMixin:
    """
    Train an SVC model and save it to the specified path.

    Parameters
    ----------
    pos_samples_features : List[np.ndarray]
        List of positive samples features.
    neg_samples_features : List[np.ndarray]
        List of negative samples features.
    model_path : str
        Path to save the model to.

    Returns
    -------
    model : ClassifierMixin
        Trained model.
    """
    X = np.concatenate([pos_samples_features, neg_samples_features])
    y = np.concatenate([np.ones(len(pos_samples_features)), np.zeros(len(neg_samples_features))])

    clf = make_pipeline(StandardScaler(), SVC(gamma="auto"))
    clf.fit(X, y)

    with open(model_path, "wb") as file:
        pickle.dump(clf, file)

    return clf
