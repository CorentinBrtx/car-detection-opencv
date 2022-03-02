# Car Detection with OpenCV and Sklearn

Car detection algorithm with classical computer vision (no deep learning) using OpenCV and Sklearn.

## Installation

To install the required dependencies, run the following command:

    pip install -r requirements.txt

## Training

To train a new model on a set of labelled images, run the following command:

    python -m src.training.train annotations.csv model.pkl

The annotation_file is a CSV file with the following columns:

    frame_id, bounding_boxes

where:

    frame_id: path to an image
    bounding_boxes: string describing the bounding boxes, in the format: x1 y1 w1 h1 x2 y2 w2 h2 ...

## Testing

You can use the following command to test your model on one image:

    python -m src.detection.detect image.jpg model.pkl output.txt

Or you can use the following command to generate a submission file for the Kaggle Car Detection competition:

    python -m src.detection.make_submission_file test_folder model.pkl submission.csv

An already trained model is provided in the `models/` folder.
