import argparse
import logging
import os
import sys
from typing import Any, Dict, Tuple

import pandas as pd
import tensorflow as tf


def split_features_labels(
    dframe: pd.DataFrame, label_col: str = "duration"
) -> Tuple[pd.Series, pd.DataFrame]:
    """Features/label splitting

    Provides two dframes (features, label) from single input dframe

    :param dframe: input pandas dataframe
    :param label_col: label column name, defaults to "duration"
    :return: (features dframe, label dframe)

    """

    labels = dframe[label_col]
    features = dframe.drop(columns=[label_col])

    return features, labels


def run(options: Dict[str, Any]) -> None:
    """Main loop

    :param options: input options as dict
    :return: None

    """
    with tf.io.gfile.GFile(options["train_dataset"]) as f:
        train_data = pd.read_csv(f)

    with tf.io.gfile.GFile(options["eval_dataset"]) as f:
        eval_data = pd.read_csv(f)

    train_data, train_labels = split_features_labels(train_data)
    eval_data, eval_labels = split_features_labels(eval_data)

    train_size = len(train_data)
    X_train = tf.data.Dataset.from_tensor_slices(train_data)
    y_train = tf.data.Dataset.from_tensor_slices(train_labels)
    X_train = X_train.batch(options["batch_size"]).repeat()
    y_train = y_train.batch(options["batch_size"]).repeat()

    X_eval = tf.data.Dataset.from_tensor_slices(eval_data)
    y_eval = tf.data.Dataset.from_tensor_slices(eval_labels)
    X_eval = X_eval.batch(options["batch_size"])
    y_eval = y_eval.batch(options["batch_size"])

    # Build your model
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=options["export_path"]
    )
    model = tf.keras.Sequential(name="bike_predict")
    model.add(
        tf.keras.layers.Dense(64, input_dim=len(train_data.iloc[0]), activation="relu")
    )
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(1))

    # Compile the model and see a summary
    optimizer = tf.keras.optimizers.Adam(0.001)
    model.compile(loss="mean_squared_logarithmic_error", optimizer=optimizer)
    model.summary()

    model.fit(
        tf.data.Dataset.zip((X_train, y_train)),
        steps_per_epoch=train_size // options["batch_size"],
        epochs=options["epochs"],
        validation_data=tf.data.Dataset.zip((X_eval, y_eval)),
        callbacks=[tensorboard_callback],
    )

    exported_model_folder = os.path.join(options["export_path"], "model")
    logging.info("saving model to: {}".format(exported_model_folder))
    model.save(exported_model_folder)


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
