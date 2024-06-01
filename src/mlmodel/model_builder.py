import tensorflow as tf
import io
import contextlib
import pickle
import gzip
import base64

from src.repository.model.model_track_repository import get_model_track_record
from src.util import log

logger = log.init_logger()


def capture_model_summary(model):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        model.summary()
    summary_str = stream.getvalue()
    return summary_str


def load_model_from_json_string(model_json: str):
    try:
        model = tf.keras.models.model_from_json(model_json)
        model_summary = capture_model_summary(model)
        logger.info("Model architecture loaded successfully.\nModel Summary:\n{0}".format(model_summary))
        return model
    except Exception as e:
        logger.error(f"Error loading model from JSON: {e}")
        raise


def compress_weights(weights):
    logger.info("Compressing weights...")
    weights_serialized = pickle.dumps(weights)
    weights_compressed = gzip.compress(weights_serialized)
    weights_encoded = base64.b64encode(weights_compressed).decode('utf-8')
    logger.info(f"Weights compressed and encoded: {weights_encoded[:100]}...")  # Log first 100 characters
    return weights_encoded


def decompress_weights(weights_encoded):
    logger.info(f"Decompressing weights. Input type: {type(weights_encoded)}, size: {len(weights_encoded)}")
    try:
        # Step 1: Decode from base64
        weights_compressed = base64.b64decode(weights_encoded)
        logger.debug(f"Decoded weights. Type: {type(weights_compressed)}, size: {len(weights_compressed)}")

        # Step 2: Decompress using gzip
        weights_serialized = gzip.decompress(weights_compressed)
        logger.info(f"Decompressed weights. Type: {type(weights_serialized)}, size: {len(weights_serialized)}")

        # Step 3: Deserialize using pickle
        weights = pickle.loads(weights_serialized)
        logger.debug("Weights decompressed and deserialized successfully.")
        return weights
    except Exception as e:
        logger.error(f"Error during decompression: {e}")
        raise


def build_model(domain_type):
    logger.info("get model from cache: {0}".format(domain_type))
    model_track_record = get_model_track_record(domain_type)
    model = load_model_from_json_string(model_track_record[0])
    logger.info("success loaded model: {0}".format(domain_type))
    # Building the model
    return model
