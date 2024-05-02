import argparse
import os
# # from src.client.flwr_client import create_client
# from src.client.onyx_flwr_client import create_client
# from src.client.flwr_client import create_client as flwr_create_client
# import flwr as fl
# from flwr.client import start_client
from src.util import log

logger = log.init_logger()
# def parse_args():
#     parser = argparse.ArgumentParser(description="Run Flower client")
#     parser.add_argument(
#         "--node-id",
#         type=int,
#         choices=[0, 1, 2],
#         required=True,
#         help="Partition of the dataset (0, 1, or 2)."
#     )
#     return parser.parse_args()

from client import client_fn, FlowerClient  # Import necessary functions from client.py
from flwr.client import start_client

def main():

    # Make TensorFlow log less verbose
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # args = parse_args()
    # try:
    #
    #     # Create and start the Flower client
    #     logger.info(" Create and start the Flower client ...")
    #     client = create_client()
    #     logger.info("Onyx Federated Learning Agent starting ...")
    #     start_client(
    #         server_address="127.0.0.1:8080",
    #         client=client.to_client()
    #     )
    #     logger.info("Onyx Flower client setup completed")
    #
    # except Exception as e:
    #     logger.error("Failed to start the onyx flwr client: %s", str(e))
    try:
        app = ClientApp(
            client_fn=client_fn,
        )
        app.start()

        # If you're using legacy mode
        start_client(
            server_address="127.0.0.1:8080",
            client=FlowerClient().to_client(),
        )
    except Exception as e:
        logger.error("Failed to start the flwr client: %s", str(e))

if __name__ == "__main__":
    main()
