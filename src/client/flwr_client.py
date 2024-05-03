from flwr.client import NumPyClient
import tensorflow as tf

class FlowerClient(NumPyClient):
    def __init__(self, cid, model, x_train, y_train, x_test, y_test):
        self.cid = cid
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return self.model.get_weights()

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}

    def predict(self, new_data):
        print(f"[Client {self.cid}] predict")
        predictions = self.model.predict(new_data)
        return predictions