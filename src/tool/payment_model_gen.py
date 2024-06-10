import tensorflow as tf


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(27,)),  # Fixed input_shape to be a tuple
        tf.keras.layers.Dense(32, activation='tanh'),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(2, activation="softmax")
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


# Build the model
model = build_model()

# Convert the model to JSON
model_json = model.to_json()

print(model_json)
