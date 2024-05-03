import argparse

from flwr.client import start_client

from src.client.flwr_client import FlowerClient
from src.util import log
logger = log.init_logger()

from src.ml.flwr_machine_learning import setup_and_load_data


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
    print("File path:", file_path)
    # Instantiate FlwrMachineLearning class
    # Setup TensorFlow and load data
    model, x_train, y_train, x_test, y_test = setup_and_load_data(args.partition_id, file_path)
    # Generate client ID
    client_id = f"client_{args.partition_id}"

    # Create a Flower client instance
    flower_client = FlowerClient(client_id, model, x_train, y_train, x_test, y_test)
    print(f"Client ID: {flower_client.cid}")
    # Assuming x_test or another set of new_data for predictions
    predictions = flower_client.predict(x_test)
    print("Predictions:", predictions)

    # Start the Flower client
    try:
        logger.info("Starting the Flower client...")
        start_client(server_address="127.0.0.1:8080", client=flower_client.to_client())
        logger.info("Flower client setup completed")
    except Exception as e:
        logger.error(f"Failed to start the Flower client: {str(e)}")


if __name__ == "__main__":
    main()