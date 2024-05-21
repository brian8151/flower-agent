from src.mlmodel.model_builder import load_model_from_json_string, compress_weights
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

    def get_model_weights(self, model_json: str):
        try:
            model = load_model_from_json_string(model_json)
            weights = model.get_weights()
            logger.info("get model weight: {0}".format(weights))
            return weights
        except Exception as e:
            logger.error(f"Error getting model weights: {e}")
            raise

    def get_model_weights_with_serialize(self, model_json: str):
        try:
            model_weights = self.get_model_weights(model_json)
            # Convert numpy arrays to lists for JSON serialization
            weights_serializable = [w.tolist() for w in model_weights]
            return weights_serializable
            return weights
        except Exception as e:
            logger.error(f"Error getting model weights with serialize: {e}")
            raise

    def get_model_weights_with_compression(self, model_json: str):
        try:
            model_weights = self.get_model_weights(model_json)
            # Compress and encode weights
            weights_compressed = compress_weights(model_weights)
            return weights_compressed
        except Exception as e:
            logger.error(f"Error getting model weights with compression: {e}")
            raise

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
