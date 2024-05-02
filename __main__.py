import argparse

from flwr.client import start_client
from src.util import log
logger = log.init_logger()

from flwr.client import ClientApp, NumPyClient
from src.client.flwr_agent import setup_and_load_data

def main():
    # Parse arguments to get partition ID
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition-id",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Partition of the dataset (0, 1, or 2).",
    )
    args, _ = parser.parse_known_args()

    # Setup TensorFlow and load data
    model, x_train, y_train, x_test, y_test = setup_and_load_data(args.partition_id)

    # Define Flower client
    class FlowerClient(NumPyClient):
        def get_parameters(self, config):
            return model.get_weights()

        def fit(self, parameters, config):
            model.set_weights(parameters)
            model.fit(x_train, y_train, epochs=1, batch_size=32)
            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)
            return loss, len(x_test), {"accuracy": accuracy}

    def client_fn(cid: str):
        """Create and return an instance of Flower `Client`."""
        return FlowerClient().to_client()

    # Flower ClientApp
    app = ClientApp(
        client_fn=client_fn,
    )

    # Start the Flower client
    try:
        logger.info("Starting the Flower client...")
        start_client(
            server_address="127.0.0.1:8080",
            client=FlowerClient().to_client(),
        )
        logger.info("Flower client setup completed")
    except Exception as e:
        logger.error("Failed to start the Flower client: %s", str(e))

if __name__ == "__main__":
    main()