import argparse
from unittest.mock import MagicMock
from src.util import log
logger = log.init_logger()
from flwr.server.strategy.fedavg import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import Code, FitRes, Status
from src.ml.flwr_machine_learning import setup_and_load_data
from src.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from src.protocol.simple_message import send_message, receive_message
from src.common.type import List, Tuple, Union
# In-memory "database"
memory_db = {}
message_queue = []
def fit(parameters, model, x_train, y_train, x_test, y_test):
    model.set_weights(parameters)
    model.fit(x_train, y_train, epochs=1, batch_size=32)
    return model.get_weights(), len(x_train), {}

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
    fit_weights, x_train_lenth = fit(parameters_from_db, model1, x_train1, y_train1, x_test1, y_test1)
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
        results: List[Tuple[ClientProxy, FitRes]] = [
            (
                MagicMock(),
                FitRes(
                    status=Status(code=Code.OK, message="Success"),
                    parameters=agg_parameters,
                    num_examples=1,
                    metrics={},
                ),
            ),
            # (
            #     MagicMock(),
            #     FitRes(
            #         status=Status(code=Code.OK, message="Success"),
            #         parameters=ndarrays_to_parameters([weights1_0, weights1_1]),
            #         num_examples=5,
            #         metrics={},
            #     ),
            # ),
        ]
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
        parameters_aggregated, metrics_aggregated =fedavg.aggregate_fit(1, results, failures)
        print(f"parameters_aggregated {parameters_aggregated}")
        print(f"metrics_aggregated {metrics_aggregated}")

if __name__ == "__main__":
    main()