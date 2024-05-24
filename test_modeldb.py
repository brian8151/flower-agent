import tensorflow as tf
import numpy as np
import modeldb
from modeldb.client import Client
from modeldb.project import Project
from modeldb.experiment import Experiment
from modeldb.run import Run

# Step 1: Define your model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(27,)),
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

# Step 2: Initialize ModelDB Client
mdb_client = Client(
    host='localhost',  # Update with your ModelDB host
    port=6543,  # Update with your ModelDB port
    database='modeldb'
)

# Step 3: Create a Project, Experiment, and Run
project = Project(name='MyProject', description='Sample Project')
experiment = Experiment(name='MyExperiment', project_id=project.id, description='Sample Experiment')
run = Run(experiment_id=experiment.id, description='Sample Run')

# Step 4: Prepare your data
data = [
    {"features": [0, 0, 0, 0, 2992.44, 405.83, 0.77, 1, 28, 45, -1, 0, 0, 0, 0, -260.72, 105.69, 0.84, 1, 1, 80, 2, 0, 5, 3, 149, 41]},
    {"features": [0, 0, 0, 0, 1893.68, 423.34, 0.92, 1, 30, 31, -1, 0, 0, 0, 0, -4431.72, 364.18, 0.91, 1, 22, 66, 2, 0, 3, 1, 57, 140]}
]
X = np.array([item["features"] for item in data])
y = np.array([0, 1])  # Dummy labels for training purposes

# Step 5: Build and compile the model
model = build_model()

# Log model architecture
run.log_model('model_architecture', model.to_json())

# Step 6: Train the model and log training metrics
history = model.fit(X, y, epochs=10, batch_size=1)

# Log training metrics
for epoch, accuracy in enumerate(history.history['accuracy']):
    run.log_metric('accuracy', accuracy, epoch)
for epoch, loss in enumerate(history.history['loss']):
    run.log_metric('loss', loss, epoch)

# Step 7: Make predictions and log them
new_data = [
    {"features": [0, 0, 0, 0, 2500.44, 400.83, 0.85, 1, 29, 44, -1, 0, 0, 0, 0, -300.72, 110.69, 0.82, 1, 2, 78, 2, 0, 4, 2, 148, 40]},
    {"features": [0, 0, 0, 0, 1800.68, 420.34, 0.89, 1, 32, 30, -1, 0, 0, 0, 0, -4500.72, 360.18, 0.90, 1, 20, 64, 2, 0, 2, 1, 55, 138]}
]
X_new = np.array([item["features"] for item in new_data])

# Make predictions
predictions = model.predict(X_new)

# Log predictions
run.log_artifact('predictions', predictions.tolist())

# Step 8: Save the model and log the model file
model.save('model.h5')
run.log_artifact('model_file', 'model.h5')

# Close the run
run.close()

# Optional: Close the client when done
mdb_client.close()