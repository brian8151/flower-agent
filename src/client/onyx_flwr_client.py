import os
import tensorflow as tf
import flwr as fl

from src.client.data_processor import DataProcessor
from src.client.machine_learning import MachineLearning

# Function to create and return the Flower client
def create_client(node_id):
    # Load model and data (MobileNetV2, CIFAR-10)
    data_processor = DataProcessor()
    payment_np_array = data_processor.fetch_and_prepare_payment_data();
    machine_learning = MachineLearning(4, 32, 64, 2)
    model = machine_learning.get_model()
    y_hat = machine_learning.predict(payment_np_array)
    data_processor.save_prediction_results(y_hat)
    x, y = data_processor.get_fit_data()

    # Download and partition dataset


    # Define Flower client
    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return model.get_weights()

        def fit(self, parameters, config):
            # model.set_weights(parameters)
            machine_learning.train_model(x, y)
            return model.get_weights(), len(x), {}

        def evaluate(self, parameters, config):
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x, y)
            return loss, len(x), {"accuracy": accuracy}

    return CifarClient()