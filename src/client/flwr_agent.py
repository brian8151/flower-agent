import os
import tensorflow as tf
from flwr_datasets import FederatedDataset
def setup_and_load_data(partition_id):
    # Make TensorFlow log less verbose
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Load model and data (MobileNetV2, CIFAR-10)
    model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Download and partition dataset
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": 3})
    partition = fds.load_partition(partition_id, "train")
    partition.set_format("numpy")

    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2, seed=42)
    x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
    x_test, y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]

    return model, x_train, y_train, x_test, y_test
