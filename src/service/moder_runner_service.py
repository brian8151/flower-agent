from src.mlmodel.model_builder import build_model_from_config
from src.mlmodel.payment.model import get_payment_config
from src.util import log

logger = log.init_logger()


class ModelRunner:
    """ Class for machine learning service """

    def __init__(self):
        pass

    def get_model_config(self, domain_type):
        if domain_type == "payment":
            return get_payment_config()
        else:
            raise ValueError(f"Unsupported domain type: {domain_type}")

    def build_model(self, domain_type):
        config = self.get_model_config(domain_type)
        # Building the model
        model = build_model_from_config(config)
        return model

    def run_mode_prediction(self, workflow_trace_id, domain_type, data, weights=None):
        logger.info("Build model for domain {0}, workflow_trace_id: {1}".format(domain_type, workflow_trace_id))
        model = self.build_model(domain_type)
        logger.info("Model summary: {0}".format(model.summary()))

        if weights is None:
            logger.info("Weight is empty, initializing weight")
            weights = model.get_weights()

        model.set_weights(weights)
        y_hat = model.predict([item.features for item in data])
        n = len(data)
        logger.info("Data size: {0}".format(n))

        data_req = [{"data": data[i].features, "result": None} for i in range(n)]

        for i in range(n):
            data_req[i]["result"] = float(100.0 * y_hat[i][0])  # acceptable percentage

        return data_req
