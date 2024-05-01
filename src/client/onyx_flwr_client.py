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
    logger.info("Initializing client creation process")

    data_processor = DataProcessor()

    # Fetch and prepare payment data, and initialize model for this data
    x, y = data_processor.fetch_and_prepare_payment_data()
    input_shape = x.shape[1]  # Dynamically determine the number of features in x
    logger.info(f"Data shape before prediction: X: {x.shape}, Y: {y.shape}")
    machine_learning = MachineLearning(input_shape, 32, 64, 2)  # Initialize model based on input shape
    model = machine_learning.get_model()
    logger.info(f"Model for {input_shape} features initialized successfully.")

    # Prediction and result saving
    y_hat = machine_learning.predict(x)
    data_processor.save_prediction_results(y_hat)
    logger.info("Predictions made successfully.")
    logger.info(f"Predictions: {y_hat}")

    # Fetch and prepare data for training, and initialize model for this data
    x, y = data_processor.get_fit_data()
    input_shape_fit = x.shape[1]  # Dynamically determine the number of features in x for training
    logger.info(f"Data for training fetched. X shape: {x.shape}, Y shape: {y.shape}")
    machine_learning_fit = MachineLearning(input_shape_fit, 32, 64,
                                           2)  # Initialize model based on input shape for fit data
    model_fit = machine_learning_fit.get_model()
    logger.info(f"Model for {input_shape_fit} features initialized successfully for training.")

    # Depending on your workflow, you might use 'model' or 'model_fit' with the corresponding 'x, y'
    # Here I assume you continue with 'model_fit' for simplicity in the Flower client setup
    return FlowerClient(model_fit, x, y)
