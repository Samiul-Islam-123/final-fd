import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import numpy as np
import threading
import json
import sys

# Global variables
AGGREGATED_MODEL_PATH = "aggregated_model.keras"
status_updates = []
client_accuracies = {}
lock = threading.Lock()


# Function to log updates
def log_update(message):
    with lock:
        status_updates.append(message)
        print(message)


# Load and preprocess the CIFAR-10 dataset
def load_dataset():
    log_update("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    log_update("Dataset loaded and normalized.")
    return x_train, y_train, x_test, y_test


# Split the dataset into `n` clients with shuffling
def split_dataset(x_train, y_train, n):
    log_update(f"Splitting dataset into {n} clients...")
    split_size = len(x_train) // n
    clients = []
    indices = np.random.permutation(len(x_train))
    x_train, y_train = x_train[indices], y_train[indices]
    for i in range(n):
        start = i * split_size
        end = start + split_size if i < n - 1 else len(x_train)
        clients.append((x_train[start:end], y_train[start:end]))
    log_update("Dataset split completed.")
    return clients


# Define different CNN architectures
def create_model(model_type):
    if model_type == "simple_cnn":
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ])
    elif model_type == "densenet":
        base_model = tf.keras.applications.DenseNet121(include_top=False, input_shape=(32, 32, 3), pooling='avg')
        base_model.trainable = False
        model = models.Sequential([
            base_model,
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ])
    elif model_type == "custom_cnn":
        model = models.Sequential([
            layers.Conv2D(64, (3, 3), activation="relu", input_shape=(32, 32, 3)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ])
    elif model_type == "resnet":
        base_model = tf.keras.applications.ResNet50(include_top=False, input_shape=(32, 32, 3), pooling='avg')
        base_model.trainable = False
        model = models.Sequential([
            base_model,
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ])
    elif model_type == "inceptionv3":
        base_model = tf.keras.applications.InceptionV3(include_top=False, input_shape=(32, 32, 3), pooling='avg')
        base_model.trainable = False
        model = models.Sequential([
            base_model,
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ])
    elif model_type == "efficientnet":
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(32, 32, 3), pooling='avg')
        base_model.trainable = False
        model = models.Sequential([
            base_model,
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ])
    elif model_type == "mobilenet":
        base_model = tf.keras.applications.MobileNet(include_top=False, input_shape=(32, 32, 3), pooling='avg')
        base_model.trainable = False
        model = models.Sequential([
            base_model,
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# Client training function
def train_client(client_id, data, result, model_type):
    try:
        x_train, y_train = data
        model = create_model(model_type)
        log_update(f"Client {client_id}: Training started with model type {model_type}.")
        
        history = model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
        training_accuracy = history.history["accuracy"][-1]

        with lock:
            client_accuracies[client_id] = training_accuracy
            result[client_id] = model.get_weights()

        log_update(f"Client {client_id}: Training completed. Accuracy: {training_accuracy:.2f}")
    except Exception as e:
        log_update(f"Client {client_id}: Training failed. Error: {str(e)}")


# Aggregation function
def aggregate_weights(client_weights):
    log_update("Aggregating models from clients...")
    total_accuracy = sum(client_accuracies.values())
    new_weights = []
    for weight_idx in range(len(client_weights[0])):
        weighted_sum = np.sum([
            client_weights[i][weight_idx] * (client_accuracies[i] / total_accuracy)
            for i in range(len(client_weights))
        ], axis=0)
        new_weights.append(weighted_sum)
    log_update("Aggregation completed.")
    return new_weights


# Save the aggregated model
def save_model(weights, model_type):
    model = create_model(model_type)
    model.set_weights(weights)
    model.save(AGGREGATED_MODEL_PATH)
    log_update(f"Aggregated model saved to {AGGREGATED_MODEL_PATH}")


# Main simulation function
def main():
    if len(sys.argv) != 3:
        print("Usage: python simulation.py <number_of_clients> <model_type>")
        sys.exit(1)

    try:
        num_clients = int(sys.argv[1])
        model_type = sys.argv[2]
        if num_clients <= 0:
            raise ValueError("Number of clients must be positive.")
    except Exception as e:
        print(f"Invalid arguments: {str(e)}")
        sys.exit(1)

    x_train, y_train, x_test, y_test = load_dataset()
    clients = split_dataset(x_train, y_train, num_clients)

    client_results = {}
    threads = []
    for i, data in enumerate(clients):
        thread = threading.Thread(target=train_client, args=(i, data, client_results, model_type))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    log_update("\nIndividual Client Accuracies:")
    for client_id, accuracy in client_accuracies.items():
        log_update(f"Client {client_id}: {accuracy:.2f}")

    client_weights = list(client_results.values())
    if client_weights:
        aggregated_weights = aggregate_weights(client_weights)
        save_model(aggregated_weights, model_type)

        model = create_model(model_type)
        model.set_weights(aggregated_weights)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        log_update(f"\nAggregated Model Accuracy: {accuracy * 100:.2f}%")
    else:
        log_update("No client models were successfully trained. Aggregation skipped.")

    with open("status_updates.json", "w") as f:
        json.dump(status_updates, f)
    with open("client_accuracies.json", "w") as f:
        json.dump(client_accuracies, f)


if __name__ == "__main__":
    main()