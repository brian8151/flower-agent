import json
import pickle

from src.cache.mem_store import setup_mem_store, create_model, get_model_data, add_weight, get_weight_by_model
from src.config import get_config
from src.mlmodel.model_builder import load_model_from_json_string
from src.protocol.http.http_utils import HttpUtils
from src.util import log

logger = log.init_logger()


class ModelOperator:
    """ Class for handle model mem store """

    def __init__(self):
        self.http_utils = HttpUtils()

    def initial_mem_store(self):
        logger.info("Initializing model mem store")
        setup_mem_store()
        logger.info("Completed setup model mem store")
        model_name_conf = get_config("app.model.name")
        logger.info("loading models: {0}".format(model_name_conf))
        model_names = [name.strip() for name in model_name_conf.split(',')]
        self.add_model(model_names)

    def add_model(self, model_names):
        # Loop through the list and process each model name
        for model in model_names:
            logger.info("adding model: {0}".format(model))
            orchestrator_client_url = get_config("app.orchestrator.client.url")
            model_url =  orchestrator_client_url + "/model/"+ model
            response = self.http_utils.call_get(model_url)
            logger.info("fetch model response: {0}".format(response))
            # Assuming the response body is in JSON format
            response_body = json.loads(response.text)
            # Convert model string to dictionary
            model_res = json.loads(response_body['model'])
            model_data = json.dumps(model_res).encode('utf-8')
            logger.info("saving model : {0}".format(model))
            # save to db
            create_model(model, model_data)
            self.add_weight(model)

    def add_weight(self, model_name):
        logger.info("saving weight model : {0}".format(model_name))
        model_id, model_data_encoded = get_model_data(model_name)
        model_json = model_data_encoded.decode('utf-8')
        model = load_model_from_json_string(model_json)
        weights = model.get_weights()
        serialized_weights = pickle.dumps(weights)  # Serialize the weights
        add_weight(model_id, serialized_weights)



