import tensorflow as tf
import io
import contextlib
import pickle
import gzip
import base64
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
    weights_compressed = base64.b64decode(weights_encoded.encode('utf-8'))
    weights_serialized = gzip.decompress(weights_compressed)
    weights = pickle.loads(weights_serialized)
    return weights
