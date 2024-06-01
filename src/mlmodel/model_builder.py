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
    weights_serialized = pickle.dumps(weights)
    weights_compressed = gzip.compress(weights_serialized)
    weights_encoded = base64.b64encode(weights_compressed).decode('utf-8')
    return weights_encoded


def decompress_weights(weights_encoded):
    if isinstance(weights_encoded, bytes):
        weights_compressed = weights_encoded
    else:
        weights_compressed = base64.b64decode(weights_encoded)

    weights_serialized = gzip.decompress(weights_compressed)
    weights = pickle.loads(weights_serialized)
    return weights


def build_model(domain_type):
    logger.info("get model from cache: {0}".format(domain_type))
    model_track_record = get_model_track_record(domain_type)
    model = load_model_from_json_string(model_track_record[0])
    logger.info("success loaded model: {0}".format(domain_type))
    # Building the model
    return model
