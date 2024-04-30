import argparse
import os
# from src.client.flwr_client import create_client
from src.client.onyx_flwr_client import create_client
import flwr as fl
from src.util import log

logger = log.init_logger()
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
    logger.info("Onyx Federated Learning Agent starting ...")
    # Make TensorFlow log less verbose
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    try:
        args = parse_args()
        client = create_client(args.node_id)
        fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
    except Exception as e:
        logger.error("Failed to start the flwr client: %s", str(e))


if __name__ == "__main__":
    main()
