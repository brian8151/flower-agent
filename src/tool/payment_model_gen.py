import tensorflow as tf


def build_model(self):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(27)),
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

model_json = build_model.to_json()

print(model_json)