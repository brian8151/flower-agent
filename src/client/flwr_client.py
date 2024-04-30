import os
import tensorflow as tf
from flwr_datasets import FederatedDataset
import flwr as fl
from src.util import log
logger = log.init_logger()
# Function to create and return the Flower client
def create_client(node_id):
    logger.info("Initializing MobileNetV2 model for CIFAR-10 dataset")
    model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    logger.info("Model compiled successfully with optimizer=adam, loss=sparse_categorical_crossentropy, metrics=['accuracy']")

    # Download and partition dataset
    logger.info("Loading CIFAR-10 dataset and partitioning")
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": 3})
    partition = fds.load_partition(node_id, "train")
    logger.info(f"Data partition loaded for node_id={node_id}")

    # Setting data format to numpy for easier handling
    partition.set_format("numpy")
    logger.info("Data format set to numpy for partition")

    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2)
    x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
    x_test, y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]
    logger.info("Data split into train and test sets")
    logger.info(f"Training data shape: {x_train.shape}, Labels: {y_train.shape}")
    logger.info(f"Testing data shape: {x_test.shape}, Labels: {y_test.shape}")

    # Define Flower client
    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            logger.info("Fetching model parameters for transmission")
            return model.get_weights()

        def fit(self, parameters, config):
            logger.info("Setting model parameters and starting training")
            model.set_weights(parameters)
            model.fit(x_train, y_train, epochs=1, batch_size=32)
            logger.info("Model training complete")
            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):
            logger.info("Evaluating model performance")
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)
            logger.info(f"Evaluation complete: Loss={loss}, Accuracy={accuracy}")
            return loss, len(x_test), {"accuracy": accuracy}

    return CifarClient()