import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import numpy as np
import threading
import flwr as fl
import os
import sys
import json

# Global variables
AGGREGATED_MODEL_PATH = "aggregated_model.keras"  # Path to save the aggregated model
status_updates = []  # List to store simulation updates
client_accuracies = {}  # Dictionary to store individual client accuracies


# Function to log updates
def log_update(message):
    status_updates.append(message)
    print(message)


# Load and preprocess the CIFAR-10 dataset
def load_dataset():
    log_update("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    log_update("Dataset loaded and normalized.")
    return x_train, y_train, x_test, y_test


# Split the dataset into `n` clients
def split_dataset(x_train, y_train, n):
    log_update(f"Splitting dataset into {n} clients...")
    split_size = len(x_train) // n
    clients = []
    for i in range(n):
        start = i * split_size
        end = start + split_size if i < n - 1 else len(x_train)
        clients.append((x_train[start:end], y_train[start:end]))
    log_update("Dataset split completed.")
    return clients


# Define a simple CNN model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# Client training function
def train_client(client_id, data, result):
    x_train, y_train = data
    model = create_model()
    log_update(f"Client {client_id}: Training started.")
    
    # Train the model
    history = model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
    
    # Save training accuracy
    training_accuracy = history.history["accuracy"][-1]
    client_accuracies[client_id] = training_accuracy
    log_update(f"Client {client_id}: Training completed. Accuracy: {training_accuracy:.2f}")
    
    # Save the model weights
    result[client_id] = model.get_weights()


# Aggregation function
def aggregate_weights(client_weights):
    log_update("Aggregating models from clients...")
    new_weights = [np.mean([client[weight] for client in client_weights], axis=0) for weight in range(len(client_weights[0]))]
    log_update("Aggregation completed.")
    return new_weights


# Save the aggregated model
def save_model(weights):
    model = create_model()
    model.set_weights(weights)
    model.save(AGGREGATED_MODEL_PATH)
    log_update(f"Aggregated model saved to {AGGREGATED_MODEL_PATH}")


# Main simulation function
def main():
    # Get the number of clients from the command line
    if len(sys.argv) != 2:
        print("Usage: python simulation.py <number_of_clients>")
        sys.exit(1)
    num_clients = int(sys.argv[1])

    # Load and split dataset
    x_train, y_train, x_test, y_test = load_dataset()
    clients = split_dataset(x_train, y_train, num_clients)

    # Train clients in separate threads
    client_results = {}
    threads = []
    for i, data in enumerate(clients):
        thread = threading.Thread(target=train_client, args=(i, data, client_results))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Log client accuracies
    log_update("\nIndividual Client Accuracies:")
    for client_id, accuracy in client_accuracies.items():
        log_update(f"Client {client_id}: {accuracy:.2f}")

    # Aggregate client weights
    client_weights = list(client_results.values())
    aggregated_weights = aggregate_weights(client_weights)

    # Save the aggregated model
    save_model(aggregated_weights)

    # Evaluate the aggregated model
    model = create_model()
    model.set_weights(aggregated_weights)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    log_update(f"\nAggregated Model Accuracy: {accuracy * 100:.2f}%")

    # Save status updates to a JSON file (for the frontend)
    with open("status_updates.json", "w") as f:
        json.dump(status_updates, f)

    # Save client accuracies to a JSON file
    with open("client_accuracies.json", "w") as f:
        json.dump(client_accuracies, f)

if __name__ == "__main__":
    main()
