# Car Detection with OpenCV and Sklearn

Car detection algorithm with classical computer vision (no deep learning) using OpenCV and Sklearn.

## Installation

To install the required dependencies, run the following command:

    pip install -r requirements.txt

## Training

To train a new model on a set of labelled images, you can use the method `train`:

    from src.training.train import train
    train(annotation_file, model_save_path)

The annotation_file is a CSV file with the following columns:

    frame_id, bounding_boxes

where:

    frame_id: path to an image
    bounding_boxes: string describing the bounding boxes, in the format: x1 y1 w1 h1 x2 y2 w2 h2 ...

## Testing

To test your model, you can use the methods in the `src/detection/` folder (`detect_cars()` in `detect.py`, or `make_submission_file()` in `make_submission.py`).

An already trained model is provided in the `models/` folder.
