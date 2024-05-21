import tensorflow as tf
import importlib

from src.util import log

logger = log.init_logger()


def load_model_from_json_string(model_json: str):
    try:
        model = tf.keras.models.model_from_json(model_json)
        logger.info("Model architecture loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model from JSON: {e}")
        raise


def build_model_from_config(config):
    model = tf.keras.Sequential(name=config['config']['name'])

    for layer_config in config['config']['layers']:
        try:
            # Dynamically import the module and class
            module = importlib.import_module(layer_config['module'])
            class_ = getattr(module, layer_config['class_name'])

            # Extract layer configuration
            layer_kwargs = layer_config['config']

            # Handle special cases such as input shape
            if layer_kwargs.get('batch_input_shape') is not None:
                layer_kwargs['batch_input_shape'] = tuple(
                    None if dim is None else dim for dim in layer_kwargs['batch_input_shape']
                )

            if layer_kwargs.get('build_config') and 'input_shape' in layer_kwargs['build_config']:
                layer_kwargs['input_shape'] = tuple(
                    None if dim is None else dim for dim in layer_kwargs['build_config']['input_shape']
                )

            # Log the current layer configuration
            logger.debug(f"Layer config before filtering: {layer_kwargs}")

            # Filter out keys that are not valid for this layer type
            valid_args = class_.__init__.__code__.co_varnames
            layer_kwargs = {k: v for k, v in layer_kwargs.items() if v is not None and k in valid_args}

            # Ensure required arguments are present
            if class_ is tf.keras.layers.Dense and 'units' not in layer_kwargs:
                raise ValueError(f"Missing required argument 'units' for {class_.__name__}")

            # Log the final layer arguments
            logger.debug(f"Layer args for {class_.__name__}: {layer_kwargs}")

            # Create layer and add to model
            layer = class_(**layer_kwargs)
            model.add(layer)
        except Exception as e:
            logger.error(f"Error adding layer {layer_config['class_name']}: {e}")
            raise

    return model
