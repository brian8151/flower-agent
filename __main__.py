import argparse

from flwr.client import start_client

from src.client.flwr_client import FlowerClient
from src.util import log
logger = log.init_logger()

from flwr.client import ClientApp, NumPyClient
from src.client.flwr_agent import setup_and_load_data
def main():
    # Parse arguments to get partition ID and CSV file name
    parser = argparse.ArgumentParser(description="Flower Client Configuration")
    parser.add_argument(
        "--partition-id",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Partition of the dataset (0, 1, or 2).",
    )
    parser.add_argument(
        "csv_file_name",
        type=str,
        help="Name of the CSV file to load data from."
    )
    args = parser.parse_args()

    # Construct the file path
    file_path = f'/apps/data/{args.csv_file_name}'

    # Setup TensorFlow and load data
    model, x_train, y_train, x_test, y_test = setup_and_load_data(args.partition_id, file_path)

    # Create a Flower client instance
    flower_client = FlowerClient(model, x_train, y_train, x_test, y_test)

    def client_fn(cid: str):
        """Create and return an instance of Flower `Client`."""
        return flower_client.to_client()

    # Flower ClientApp
    app = ClientApp(client_fn=client_fn)

    # Start the Flower client
    try:
        logger.info("Starting the Flower client...")
        start_client(server_address="127.0.0.1:8080", client=flower_client.to_client())
        logger.info("Flower client setup completed")
    except Exception as e:
        logger.error(f"Failed to start the Flower client: {str(e)}")

if __name__ == "__main__":
    main()