import argparse
import logging
import sys
from typing import Any, Dict, Tuple

import pandas as pd
import tensorflow as tf


def split_features_labels(
    dframe: pd.DataFrame, label_col: str = "duration"
) -> Tuple[pd.Series, pd.DataFrame]:

    labels = dframe[label_col]
    features = dframe.drop(columns=[label_col])

    return features, labels


def run(options: Dict[str, Any]) -> None:

    with tf.io.gfile.GFile(options["train_dataset"]) as f:
        train_data = pd.read_csv(f)

    with tf.io.gfile.GFile(options["eval_dataset"]) as f:
        eval_data = pd.read_csv(f)

    train_data, train_labels = split_features_labels(train_data)
    eval_data, eval_labels = split_features_labels(eval_data)

    # Build your model
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

    input_train = tf.data.Dataset.from_tensor_slices(train_data)
    output_train = tf.data.Dataset.from_tensor_slices(train_labels)
    input_train = input_train.batch(options["batch_size"]).repeat()
    output_train = output_train.batch(options["batch_size"]).repeat()
    train_dataset = tf.data.Dataset.zip((input_train, output_train))

    train_size = len(train_data)
    model.fit(
        train_dataset,
        steps_per_epoch=train_size // options["batch_size"],
        epochs=options["epochs"],
    )

    # Run evaluation
    results = model.evaluate(eval_data, eval_labels)
    logging.info(results)

    model.save(options["export_path"])


def build_parser() -> argparse.ArgumentParser:
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

    parser.add_argument("--batch_size", required=True, type=int, help="Batch size")

    parser.add_argument("--epochs", required=True, type=int, help="Epochs")

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
