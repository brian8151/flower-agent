import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from src.client.data_loader import DataLoader


def setup_and_load_data(partition_id, data_path, test_size=0.2, random_seed=42):
    # Create an instance of DataLoader and load data
    data_loader = DataLoader(data_path)
    features, labels = data_loader.load_data()

    # Preprocessing: Scale continuous data and encode categorical data
    column_trans = ColumnTransformer([
        ('scale', StandardScaler(), ['transaction_amount']),
        ('onehot', OneHotEncoder(), ['transaction_type', 'customer_type'])
    ], remainder='passthrough')

    features = column_trans.fit_transform(features)

    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_seed)

    # Define a simple neural network model for binary classification
    model = Sequential([
        Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model, x_train, y_train, x_test, y_test



def setup_and_load_data_old(partition_id):
    # Make TensorFlow log less verbose
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    data ="payment.csv"
    # Load model and data (MobileNetV2, CIFAR-10)
    model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Download and partition dataset
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": 3})
    partition = fds.load_partition(partition_id, "train")
    partition.set_format("numpy")

    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2, seed=42)
    x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
    x_test, y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]


    return model, x_train, y_train, x_test, y_test
