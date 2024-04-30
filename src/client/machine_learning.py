
import flwr as fl
import tensorflow as tf
import pandas as pd
import numpy as np
from src.util import log
logger = log.init_logger()

class MachineLearning:
    """ Class for machine learning service """
    def __init__(self, input_shape=4, first_layer_units=32, second_layer_units=64, output_units=2):
        logger.info(
            "Initializing Machine Learning model with input_shape=%d, first_layer_units=%d, second_layer_units=%d, output_units=%d",
            input_shape, first_layer_units, second_layer_units, output_units)
        self.model = self.build_model(input_shape, first_layer_units, second_layer_units, output_units)


    def get_model(self):
        """Returns the model instance."""
        return self.model
    def build_model(self, input_shape, first_layer_units, second_layer_units, output_units):
        """Builds and compiles a Keras model based on provided specifications."""
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_shape,)),
            tf.keras.layers.Dense(first_layer_units, activation='tanh'),
            tf.keras.layers.Dense(second_layer_units, activation='tanh'),
            tf.keras.layers.Dense(output_units, activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        logger.info(
            "Model compiled with optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']")
        return model

    # def build_model(self):
    #     # Load model and data (MobileNetV2, CIFAR-10)
    #     model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    #     model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    #     return model

    def train_model(self, x, y, epochs=10, batch_size=32):
        """Train the model with the provided data."""
        logger.info("Starting training model with epochs=%d, batch_size=%d", epochs, batch_size)
        logger.info("Training data shapes - X: %s, Y: %s", x.shape, y.shape)
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size)
        logger.info("Model training complete")

    def predict(self, x):
        """Make predictions with the model on new data."""
        logger.info("Making predictions on data of shape: %s", x.shape)
        predictions = self.model.predict(x)
        logger.info("Predictions: %s", predictions)
        return predictions



