import os
import tensorflow as tf
import flwr as fl

from src.client.data_processor import DataProcessor
from src.client.machine_learning import MachineLearning
from src.util import log

logger = log.init_logger()
# Function to create and return the Flower client


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, x, y):
        self.model = model
        self.x = x
        self.y = y

    def get_parameters(self, config):
        weights = self.model.get_weights()
        logger.info(f"Fetching model parameters for training, weights: {weights}")
        return weights

    def fit(self, parameters, config):
        logger.info(f"Setting model parameters and starting fit process with data shapes X: {self.x.shape}, Y: {self.y.shape}")
        self.model.set_weights(parameters)
        self.model.fit(self.x, self.y, epochs=1, batch_size=32)
        logger.info("Model training completed.")
        return self.model.get_weights(), len(self.x), {}

    def evaluate(self, parameters, config):
        logger.info(f"Evaluating model with data shapes X: {self.x.shape}, Y: {self.y.shape}")
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x,self.y)
        logger.info(f"Evaluation completed. Loss: {loss}, Accuracy: {accuracy}")
        return loss, len(self.x), {"accuracy": accuracy}

    logger.info("Flower client setup completed.")
def create_client():
    # Load model and data (MobileNetV2, CIFAR-10)
    logger.info(f"Initializing client creation process")
    # Load model and data (MobileNetV2, CIFAR-10)
    logger.info("Starting data preparation...")
    data_processor = DataProcessor()
    payment_np_array = data_processor.fetch_and_prepare_payment_data();
    logger.info("Data preparation completed.")
    logger.info("Initializing machine learning model...")
    machine_learning = MachineLearning(1, 32, 64, 2)
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
    return FlowerClient(model, x, y)
