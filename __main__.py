import argparse
import os
from src.client.flwr_client import create_client
import flwr as fl


def parse_args():
    parser = argparse.ArgumentParser(description="Run Flower client")
    parser.add_argument(
        "--node-id",
        type=int,
        choices=[0, 1, 2],
        required=True,
        help="Partition of the dataset (0, 1, or 2)."
    )
    return parser.parse_args()


def main():
    # Make TensorFlow log less verbose
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Parse command line arguments
    args = parse_args()

    # Create a Flower client
    client = create_client(args.node_id)

    # Start Flower client
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
