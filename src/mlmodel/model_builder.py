import tensorflow as tf
import importlib


def build_model_from_config(config):
    model = tf.keras.Sequential(name=config['config']['name'])

    for layer_config in config['config']['layers']:
        # Dynamically import the module and class
        module = importlib.import_module(layer_config['module'])
        class_ = getattr(module, layer_config['class_name'])

        # Extract layer configuration
        layer_kwargs = layer_config['config']

        # Handle special cases such as input shape
        if 'batch_input_shape' in layer_kwargs:
            layer_kwargs['batch_input_shape'] = tuple(
                None if dim is None else dim for dim in layer_kwargs['batch_input_shape']
            )

        if 'build_config' in layer_config and 'input_shape' in layer_config['build_config']:
            layer_kwargs['input_shape'] = tuple(
                None if dim is None else dim for dim in layer_config['build_config']['input_shape']
            )

        # Create layer and add to model
        layer = class_(**layer_kwargs)
        model.add(layer)

    return model

