from src.cache.mem_store import setup_mem_store, get_model
from src.config import get_config
from src.util import log

logger = log.init_logger()


class ModelOperator:
    """ Class for handle model mem store """

    def __init__(self):
        pass

    def initial_mem_store(self):
        logger.info("Initializing model mem store")
        setup_mem_store()
        logger.info("Completed setup model mem store")
        model_name_conf = get_config("app.model.name")
        logger.info("loading models: {0}".format(model_name_conf))
        model_names = [name.strip() for name in model_name_conf.split(',')]
        self.add_model(model_names)

    def add_model(self, model_names):
        print("add_model......")
        # Loop through the list and process each model name
        for model in model_names:
            print(model)
