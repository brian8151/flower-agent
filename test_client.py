import argparse
from unittest.mock import MagicMock

from src.common.onyx_custom_client_proxy import CustomClientProxy
from src.util import log
logger = log.init_logger()
from flwr.server.strategy.fedavg import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import Code, FitRes, Status, EvaluateRes, EvaluateIns
from src.ml.flwr_machine_learning import setup_and_load_data
from src.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from src.protocol.simple_message import send_message, receive_message
from flwr.common.typing import List, Tuple, Union
from flwr.common import Metrics
from flwr.server.client_manager import SimpleClientManager
# In-memory "database"
memory_db = {}
message_queue = []
def fit(parameters, model, x_train, y_train, x_test, y_test):
    model.set_weights(parameters)
    model.fit(x_train, y_train, epochs=1, batch_size=32)
    return model.get_weights(), len(x_train), {}
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    logger.info(" set up weighted_average")
    # Calculate weighted accuracies
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Log each client's accuracy and the number of examples used
    for i, (acc, ex) in enumerate(zip(accuracies, examples)):
        logger.info(f"Client {i}: Accuracy={acc / ex}, Examples={ex}")

    total_examples = sum(examples)
    total_weighted_accuracy = sum(accuracies)
    weighted_avg_accuracy = total_weighted_accuracy / total_examples
    # Log aggregate information
    logger.info(f"Total Examples: {total_examples}")
    logger.info(f"Total Weighted Accuracy: {total_weighted_accuracy}")
    logger.info(f"Weighted Average Accuracy: {weighted_avg_accuracy}")

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": weighted_avg_accuracy}


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
    # file_path = f'/apps/data/{args.csv_file_name}'
    file_path1= f'/apps/data/mock_payment_data-0.3.csv'
    print("File path1:", file_path1)
    # Instantiate FlwrMachineLearning class
    # Setup TensorFlow and load data
    model, x_train, y_train, x_test, y_test = setup_and_load_data(args.partition_id, file_path1)
    # Generate client ID
    client_id = f"client_{args.partition_id}"
    print("client_id:", client_id)
    predictions = model.predict(x_test)
    print("Predictions:", predictions)
    # Get model weights
    weights = model.get_weights()
    print("Prediction Model weights:", weights)
    parameters = ndarrays_to_parameters(weights)
    #save to db
    print("save Model weights to db :")
    # Store the parameters in the in-memory database
    memory_db['model_weights'] = parameters
    # Retrieve the parameters from the in-memory database
    print("Retrieve Model weights from db :")
    parameters_from_db = memory_db['model_weights']
    weights_from_db = parameters_to_ndarrays(parameters_from_db)
    print("DB Model weights:", weights_from_db)
    # Set the weights back to your model
    model.set_weights(weights_from_db)
    print("After feedback, we get new data set")
    file_path2= f'/apps/data/mock_payment_data-0.7.csv'
    print("File path2:", file_path2)
    # Instantiate FlwrMachineLearning class
    # Setup TensorFlow and load data
    print("rerun model")
    model1, x_train1, y_train1, x_test1, y_test1 = setup_and_load_data(args.partition_id, file_path2)
    print("now run fit")
    fit_weights, x_train_length, additional_info = fit(weights_from_db, model1, x_train1, y_train1, x_test1, y_test1)
    print("Fit Model weights:", fit_weights)
    print("send to agg with config protocol")
    # Serialize weights to send
    ser_parameters = ndarrays_to_parameters(fit_weights)
    send_message(message_queue, {"client_id": "bank1", "parameters": ser_parameters})
    # Receive and process message
    message = receive_message(message_queue)
    print("received fit_weights on another side")
    if message:
        print("Received message on another side")
        client_id = message['client_id']
        received_parameters = message['parameters']
        received_weights = parameters_to_ndarrays(received_parameters)
        agg_parameters= ndarrays_to_parameters(received_weights)
        print(f"Received weights for client {client_id}: {received_weights}")
        fedavg = FedAvg()
        cid_expected = "1"
        client_proxy = CustomClientProxy(cid=cid_expected)
        results: List[Tuple[ClientProxy, FitRes]] = [
            (
                client_proxy,
                FitRes(
                    status=Status(code=Code.OK, message="Success"),
                    parameters=agg_parameters,
                    num_examples=1,
                    metrics={},
                ),
            )
        ]
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
        print(f"fedavg.aggregate_fit --------------------->")
        parameters_aggregated, metrics_aggregated =fedavg.aggregate_fit(1, results, failures)
        print(f"check parameters_aggregated --------------------->")
        if parameters_aggregated is not None:
            print(".......................saving parameters_aggregated.......................")
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays = parameters_to_ndarrays(parameters_aggregated)
            print("saved parameters_aggregated to db DB Model weights:", aggregated_ndarrays)
            print(f"metrics_aggregated {metrics_aggregated}")
        print(f"weighted_average--------------------->")
        weighted_average(metrics_aggregated)
        # evaluate_result = fedavg.evaluate(1, agg_parameters)
        # print(f"Evaluation Result: Loss = {evaluate_result.loss}, Metrics = {evaluate_result.metrics}")
        # evaluate_res = EvaluateRes(
        #     status=Status(code=Code.OK, message="Success"), loss=evaluate_result.loss, num_examples=1, metrics=evaluate_result.metrics
        # )
        weighted_average(metrics_aggregated)
        # eval_results = []
        # eval_results.append((client_proxy, evaluate_res))
        # aggregated_result = fedavg.aggregate_evaluate(1, eval_results, [])
        # aggregated_loss, aggregated_metrics = aggregated_result
        # print(f"Aggregated Loss: {aggregated_loss}")
        # print(f"Aggregated Metrics: {aggregated_metrics}")



if __name__ == "__main__":
    main()