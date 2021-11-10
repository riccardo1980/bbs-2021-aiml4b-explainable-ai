import os
from typing import Tuple
from urllib import parse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from google.cloud import storage


def url_split(url: str) -> Tuple[str, str, str]:
    """Split url in scheme, location, path

        <scheme>://<location>[/path]

    :param str: input uri
    :return: (scheme, location, path)

    Example

    GCS blob:
    gs://flowers-public/tfrecords-jpeg-192x192-2/flowers14-230.tfrec
    ('gs', 'flowers', 'tfrecords-jpeg-192x192-2/flowers14-230.tfrec')

    Local file
    data/flowers14-230.tfrec
    ('', '', 'data/flowers14-230.tfrec')

    """
    split_results = parse.urlsplit(url)
    return (split_results.scheme, split_results.netloc, split_results.path)


def copy_blob(
    bucket_name: str,
    blob_name: str,
    destination_bucket_name: str,
    destination_blob_name: str,
) -> None:
    """Copies a blob from one bucket to another with a new name."""
    # bucket_name = "your-bucket-name"
    # blob_name = "your-object-name"
    # destination_bucket_name = "destination-bucket-name"
    # destination_blob_name = "destination-object-name"

    storage_client = storage.Client()

    source_bucket = storage_client.bucket(bucket_name)
    source_blob = source_bucket.blob(blob_name)
    destination_bucket = storage_client.bucket(destination_bucket_name)

    blob_copy = source_bucket.copy_blob(
        source_blob, destination_bucket, destination_blob_name
    )

    print(
        "Blob {} in bucket {} copied to blob {} in bucket {}.".format(
            source_blob.name,
            source_bucket.name,
            blob_copy.name,
            destination_bucket.name,
        )
    )


def files_bulk_copy(
    uri_list: Tuple[str], destination_bucket_name: str, folder: str
) -> None:

    for source_url in uri_list:
        scheme, bucket_name, blob_name = url_split(source_url)

        if scheme and bucket_name:
            if blob_name.startswith("/"):
                blob_name = blob_name[1:]

        copy_blob(
            bucket_name,
            blob_name,
            destination_bucket_name,
            os.path.join(folder, blob_name.split("/")[-1]),
        )


def dataset_to_numpy_util(
    dataset: tf.data.Dataset, N: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract images/labels from TensorFlow Dataset

        :param dataset: input dataset
        :return: (images, labels) as numpy arrays
    """

    dataset = dataset.batch(N)

    if tf.executing_eagerly():
        #  In eager mode, iterate in the Dataset directly.
        for images, labels in dataset:
            numpy_images = images.numpy()
            numpy_labels = labels.numpy()
            break
    else:
        # In non-eager mode, must get the TF note that
        # yields the nextitem and run it in a tf.Session.
        get_next_item = dataset.make_one_shot_iterator().get_next()
        with tf.Session() as sess:
            numpy_images, numpy_labels = sess.run(get_next_item)

    return numpy_images, numpy_labels


def display_one_flower(
    image: np.ndarray, title: str, subplot: int, red: bool = False
) -> int:

    """
    Visualization core util

    :param image: raw image as index map
    :param title: subplot title
    :param subplot: subplot index
    :param red: whether to print title in red
    :return: next subplot index
    """

    plt.subplot(subplot)
    plt.axis("off")
    plt.imshow(image)
    plt.title(title, fontsize=16, color="red" if red else "black")
    return subplot + 1


def title_from_label_and_target(
    label: np.ndarray, correct_label: np.ndarray, classes: Tuple[str]
) -> Tuple[str, bool]:

    """
    Title building util

    When prediction is correct:
        title contains class name, is_correct is True
    Otherwise:
        title contains both prediction class name and correct class name,
        is_correct is False

    :param label: one hot prediction label vector
    :param correct_label: one hot correct label vector
    :param classes: list of class names
    :return: (title, is_correct)
    """

    label_idx: int = np.argmax(label, axis=-1)  # one-hot to class number
    correct_label_idx: int = np.argmax(
        correct_label, axis=-1
    )  # one-hot to class number
    correct = label_idx == correct_label_idx
    return (
        "{} [{}{}{}]".format(
            classes[label_idx],
            str(correct),
            ", shoud be " if not correct else "",
            classes[correct_label_idx] if not correct else "",
        ),
        correct,
    )


def display_9_images_from_dataset(
    dataset: tf.data.Dataset, classes: Tuple[str]
) -> None:
    subplot = 331
    plt.figure(figsize=(13, 13))
    images, labels = dataset_to_numpy_util(dataset, 9)
    for i, image in enumerate(images):
        title = classes[np.argmax(labels[i], axis=-1)]
        subplot = display_one_flower(image, title, subplot)
        if i >= 8:
            break

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


def display_9_images_with_predictions(
    images: np.ndarray, predictions: np.ndarray, labels: np.ndarray, classes: Tuple[str]
) -> None:

    """
    Visualization core util

    :param images: batch of raw images
    :param predictions: predictions as array of one-hot
    :param labels: true labels as array of one-hot
    :param classes: list of class names
    """

    subplot = 331
    plt.figure(figsize=(13, 13))
    for i, image in enumerate(images):
        title, correct = title_from_label_and_target(predictions[i], labels[i], classes)
        subplot = display_one_flower(image, title, subplot, not correct)
        if i >= 8:
            break

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
