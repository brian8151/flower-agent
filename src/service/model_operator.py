from src.cache.mem_store import setup_mem_store, get_model
from src.config import get_config
from src.protocol.http.http_utils import HttpUtils
from src.util import log
import json
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
            response = self.http_utils.call_get(orchestrator_client_url)
            logger.info("model response: {0}".format(response))
            # Assuming the response body is in JSON format
            response_body = json.loads(response.text)
            model_res = response_body.model
            print(model_res)
