from typing import List, Tuple

from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    # Initialize lists to hold calculated accuracies and number of examples
    accuracies = []
    examples = []

    # Print header for detailed metrics
    print("Detailed Client Metrics:")
    print("Client | Examples | Accuracy")

    # Iterate through each client's metrics
    for i, (num_examples, m) in enumerate(metrics):
        client_accuracy = m["accuracy"]
        accuracies.append(num_examples * client_accuracy)
        examples.append(num_examples)

        # Print each client's number of examples and accuracy
        print(f"Client {i+1} | {num_examples} | {client_accuracy:.4f}")


    # Calculate weighted average accuracy
    weighted_avg_accuracy = sum(accuracies) / sum(examples)

    # Print the weighted average accuracy
    print(f"Weighted Average Accuracy: {weighted_avg_accuracy:.4f}")

    # Return custom metric (weighted average)
    return {"accuracy": weighted_avg_accuracy}

# Define strategy
strategy = FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

# Define config
config = ServerConfig(num_rounds=3)

# Flower ServerApp
app = ServerApp(
    config=config,
    strategy=strategy,
)

# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server

    start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
    )
