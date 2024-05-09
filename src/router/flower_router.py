from fastapi import APIRouter

from src.common.parameter import ndarrays_to_parameters
from src.model.client_message_res import ClientMessageResponse
from src.model.message_req import MessageRequest
from src.service.flower.flower_fedavg_service import FlowerFedAvgService
flower_router = APIRouter()
from src.util import log
logger = log.init_logger()
from src.ml.flwr_machine_learning import setup_and_load_data
def fit(parameters, model, x_train, y_train, x_test, y_test):
    model.set_weights(parameters)
    model.fit(x_train, y_train, epochs=1, batch_size=32)
    return model.get_weights(), len(x_train), {}

def client_evaluate(model, parameters, x_test, y_test):
    print(f"---- client_evaluate-----")
    model.set_weights(parameters)
    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, len(x_test), {"accuracy": accuracy}

@flower_router.post("/send-fedavg")
async def process_fed_avg(message: MessageRequest):
    client_id = message.client_id
    # Log or process the received data
    logger.info("Received from client {0}: ".format(client_id))
    flower_fed_avg_svc = FlowerFedAvgService()
    file_path = f'/apps/data/mock_payment_data-0.7.csv'
    print("File path:", file_path)
    # Instantiate FlwrMachineLearning class
    # Setup TensorFlow and load data
    print("rerun model")
    model, x_train, y_train, x_test, y_test = setup_and_load_data(file_path)
    weights = model.get_weights()
    print("Prediction Model weights:", weights)
    print("now run fit")
    fit_weights, x_train_length, additional_info = fit(weights, model, x_train, y_train, x_test, y_test)
    print("Fit Model weights:", fit_weights)
    loss, num_examples, metrics = client_evaluate(model, weights, x_test, y_test);
    # Print or use the results
    print(f"Loss: {loss}")
    print(f"Number of Test Examples: {num_examples}")
    print(f"Metrics: {metrics}")
    # Serialize the model weights to send
    ser_parameters = ndarrays_to_parameters(weights)
    # Prepare and send the message containing weights and metrics
    return ClientMessageResponse(
        message_id=message.message_id,
        client_id=message.client_id,
        strategy=message.strategy,
        parameters=ser_parameters,
        metrics=metrics,
        num_examples=num_examples,
        loss=loss,
        properties={"additional_info": additional_info}
    )
