from src.mlmodel.model_builder import load_model_from_json_string, compress_weights, build_model
from src.repository.model.model_track_repository import get_model_track_record, create_model_track_records
from src.util import log

logger = log.init_logger()


class ModelRunner:
    """ Class for machine learning service """

    def __init__(self):
        pass

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

    def initial_weights(self, domain, model_version, model_json: str):
        """
        Initialize weights for the given domain. If no global model track exists for the domain,
        it creates an entry with the same model and weight.

        Parameters:
            domain (str): The domain for which to initialize weights.
            model_version (str): The version of the model.
            model_json (str): The JSON representation of the model.

        Returns:
            str: Compressed and encoded model weights.

        Raises:
            Exception: If there is an error in getting or processing model weights.
        """
        try:
            model_track_record = get_model_track_record(domain)
            if not model_track_record:
                logger.info(f"No global model track found for domain '{domain}'. Creating a new entry.")
                local_weights_version = 1
                model_weights = self.get_model_weights(model_json)
                # Compress and encode weights
                weights_compressed = compress_weights(model_weights)
                create_model_track_records(domain, model_json, model_version, domain, weights_compressed, local_weights_version)
                return weights_compressed
            else:
                local_model_weights = model_track_record[2]
                return local_model_weights
        except Exception as e:
            logger.error(f"Error getting model weights with compression: {e}")
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

    def run_model_prediction(self, workflow_trace_id, domain_type, data):
        logger.info("Build model for domain {0}, workflow_trace_id: {1}".format(domain_type, workflow_trace_id))
        model = build_model(domain_type)
        logger.info("Model summary: {0}".format(model.summary()))
        model_track_record = get_model_track_record(domain_type)
        local_model_weights = model_track_record[2]
        global_model_weights = model_track_record[4]
        if global_model_weights is None:
            logger.info("global_model_weights is empty, use default local model weight")
            weights = local_model_weights
        else:
            logger.info("found global_model_weights")
            weights = global_model_weights

        model.set_weights(weights)
        y_hat = model.predict([item.features for item in data])
        n = len(data)
        logger.info("Data size: {0}".format(n))

        data_req = [{"data": data[i].features, "result": None} for i in range(n)]

        for i in range(n):
            data_req[i]["result"] = float(100.0 * y_hat[i][0])  # acceptable percentage

        return data_req
