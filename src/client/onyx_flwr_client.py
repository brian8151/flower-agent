import os
import tensorflow as tf
import flwr as fl

from src.client.data_processor import DataProcessor
from src.client.machine_learning import MachineLearning
from src.util import log

logger = log.init_logger()
# Function to create and return the Flower client
def create_client(node_id):
    logger.info(f"Initializing client creation process for node ID: {node_id}")
    # Load model and data (MobileNetV2, CIFAR-10)
    logger.info("Starting data preparation...")
    data_processor = DataProcessor()
    payment_np_array = data_processor.fetch_and_prepare_payment_data();
    logger.info("Data preparation completed.")
    logger.info("Initializing machine learning model...")
    machine_learning = MachineLearning(4, 32, 64, 2)
    model = machine_learning.get_model()
    logger.info("Model initialized successfully.")
    # Prediction and result saving
    logger.info("Making predictions on the prepared data...")
    y_hat = machine_learning.predict(payment_np_array)
    logger.info("Predictions made successfully.")
    logger.info(f"Predictions: {y_hat}")

    data_processor.save_prediction_results(y_hat)
    logger.info("Prediction results saved successfully.")
    x, y = data_processor.get_fit_data()
    logger.info(f"Data for training fetched. X shape: {x.shape}, Y shape: {y.shape}")
    # Download and partition dataset


    # Define Flower client
    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            logger.info("Fetching model parameters for training...")
            return model.get_weights()

        def fit(self, parameters, config):
            logger.info("Setting model parameters and starting fit process...")
            # model.set_weights(parameters)
            model.fit(x, y, epochs=10, batch_size=32)  # Adjust epochs and batch_size as needed
            logger.info("Model training completed.")
            return model.get_weights(), len(x), {}

        def evaluate(self, parameters, config):
            logger.info("Evaluating model...")
            # model.set_weights(parameters)
            loss, accuracy = model.evaluate(x, y)
            logger.info(f"Evaluation completed. Loss: {loss}, Accuracy: {accuracy}")
            return loss, len(x), {"accuracy": accuracy}

    logger.info("Flower client setup completed.")
    return CifarClient()