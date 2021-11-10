import argparse
import logging
import os
import sys
from typing import Any, Callable, Dict, Tuple

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    MaxPooling2D,
)
from tensorflow.keras.optimizers import Adam


def build_model(image_size: Tuple[int, int], n_classes: int = 5) -> tf.keras.Model:
    """Build a basic CNN for classification"""
    model = Sequential(
        [
            # Stem
            Conv2D(
                kernel_size=3,
                filters=16,
                padding="same",
                activation="relu",
                input_shape=[*image_size, 3],
            ),
            BatchNormalization(),
            Conv2D(kernel_size=3, filters=32, padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=2),
            # Conv Group
            Conv2D(kernel_size=3, filters=64, padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=2),
            Conv2D(kernel_size=3, filters=96, padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=2),
            # Conv Group
            Conv2D(kernel_size=3, filters=128, padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=2),
            Conv2D(kernel_size=3, filters=128, padding="same", activation="relu"),
            BatchNormalization(),
            # 1x1 Reduction
            Conv2D(kernel_size=1, filters=32, padding="same", activation="relu"),
            BatchNormalization(),
            # Classifier
            GlobalAveragePooling2D(),
            Dense(n_classes, activation="softmax"),
        ]
    )
    return model


def load_dataset(filenames: Tuple[str], read_tfrecord: Callable) -> tf.data.Dataset:
    # Read data from TFRecords
    AUTO = tf.data.experimental.AUTOTUNE

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.interleave(
        tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO
    )  # faster
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
    return dataset


def get_batched_dataset(
    filenames: Tuple[str],
    read_function: Callable,
    batch_size: int = 64,
    autotune_buffer_size: int = tf.data.experimental.AUTOTUNE,
    shuffle_buffer_size: int = 2048,
) -> tf.data.Dataset:

    dataset = load_dataset(filenames, read_function)
    # dataset = dataset.cache()  # This dataset fits in RAM
    # dataset = dataset.repeat()
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(autotune_buffer_size)

    return dataset


def get_read_tfrecord(image_size: Tuple[int, int], n_classes: int) -> Callable:
    def read_function(example: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        features = {
            "image": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
            "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means scalar
            "one_hot_class": tf.io.VarLenFeature(tf.float32),
        }
        example = tf.io.parse_single_example(example, features)
        image = tf.image.decode_image(
            example["image"], channels=3, expand_animations=False
        )
        image = (
            tf.cast(image, tf.float32) / 255.0
        )  # convert image to floats in [0, 1] range
        image = tf.reshape(
            image, [*image_size, 3]
        )  # explicit size will be needed for TPU
        one_hot_class = tf.sparse.to_dense(example["one_hot_class"])
        one_hot_class = tf.reshape(one_hot_class, [n_classes])
        return image, one_hot_class

    return read_function


def run(options: Dict[str, Any]) -> None:
    """Main loop

    :param options: input options as dict
    :return: None

    """

    image_size = (192, 192)
    n_classes = 5

    train_filenames = tf.io.gfile.glob(
        os.path.join(options["train_dataset"], "*.tfrec"),
    )
    train_dataset = get_batched_dataset(
        train_filenames,
        batch_size=options["batch_size"],
        read_function=get_read_tfrecord(image_size, n_classes),
    )

    validation_filenames = tf.io.gfile.glob(
        os.path.join(options["eval_dataset"], "*.tfrec")
    )
    validation_dataset = get_batched_dataset(
        validation_filenames,
        batch_size=options["batch_size"],
        read_function=get_read_tfrecord(image_size, n_classes),
    )

    model = build_model(image_size=image_size, n_classes=n_classes)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=options["export_path"]
    )

    model.compile(
        optimizer=Adam(lr=0.005, decay=0.98),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    model.fit(
        train_dataset,
        epochs=options["epochs"],
        validation_data=validation_dataset,
        callbacks=[tensorboard_callback],
    )

    def _preprocess(bytes_input: tf.TensorSpec) -> tf.TensorSpec:
        decoded = tf.io.decode_image(bytes_input, channels=3, expand_animations=False)
        decoded = tf.image.convert_image_dtype(decoded, tf.float32)
        resized = tf.image.resize(decoded, size=(192, 192))
        return resized

    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def preprocess_fn(bytes_inputs: tf.TensorSpec) -> Dict[str, tf.TensorSpec]:
        with tf.device("cpu:0"):
            decoded_images = tf.map_fn(_preprocess, bytes_inputs, dtype=tf.float32)
        return {
            "numpy_inputs": decoded_images
        }  # User needs to make sure the key matches model's input

    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def serving_fn(bytes_inputs: tf.TypeSpec) -> tf.TensorSpec:
        images = preprocess_fn(bytes_inputs)
        prob = m_call(**images)
        return prob

    exported_model_folder = os.path.join(options["export_path"], "model")
    logging.info("saving model to: {}".format(exported_model_folder))

    m_call = tf.function(model.call).get_concrete_function(
        [
            tf.TensorSpec(
                shape=[None, 192, 192, 3], dtype=tf.float32, name="numpy_inputs"
            )
        ]
    )

    tf.keras.models.save_model(
        model,
        exported_model_folder,
        signatures={
            "serving_default": serving_fn,
            "xai_preprocess": preprocess_fn,  # Required for XAI
            "xai_model": m_call,  # Required for XAI
        },
    )


def build_parser() -> argparse.ArgumentParser:
    """Parser building logic

    :return: parser

    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_dataset", required=True, type=str, help="train dataset"
    )

    parser.add_argument(
        "--eval_dataset", required=False, type=str, default=None, help="eval dataset"
    )

    parser.add_argument(
        "--export_path", required=True, type=str, help="Model destination"
    )

    parser.add_argument("--batch_size", type=int, help="Batch size", default=256)

    parser.add_argument("--epochs", type=int, help="Epochs", default=3)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    parsed_args, _ = parser.parse_known_args(sys.argv[1:])
    options = vars(parsed_args)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for key, value in options.items():
        logging.info("{}: {}".format(key, value))

    run(options)
