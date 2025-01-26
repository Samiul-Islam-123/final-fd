import numpy as np
import tensorflow as tf
import threading
import sys
import json

# Helper functions
def log_update(message):
    print(message)

def load_dataset():
    # Load your dataset here. For example:
    # - Replace with actual dataset loading logic
    # - Split dataset into x_train, y_train, x_test, y_test
    x_train, y_train, x_test, y_test = np.random.randn(1000, 784), np.random.randint(0, 2, 1000), np.random.randn(200, 784), np.random.randint(0, 2, 200)
    return x_train, y_train, x_test, y_test

def split_dataset(x_train, y_train, num_clients):
    # Split the dataset into num_clients parts
    client_data = []
    data_size = len(x_train) // num_clients
    for i in range(num_clients):
        start = i * data_size
        end = (i + 1) * data_size if i != num_clients - 1 else len(x_train)
        client_data.append((x_train[start:end], y_train[start:end]))
    return client_data

def create_model():
    # Create a simple model for federated learning
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_client(client_id, data, client_results):
    x_train, y_train = data
    log_update(f"Client {client_id}: Training started.")
    model = create_model()
    model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
    
    accuracy = model.evaluate(x_train, y_train, verbose=0)[1]
    client_results[client_id] = model.get_weights()
    log_update(f"Client {client_id}: Training completed. Accuracy: {accuracy:.2f}")
    client_accuracies[client_id] = accuracy

# Federated Learning Aggregation Function
def aggregate_weights(client_weights, client_data_sizes):
    log_update("Aggregating models from clients...")
    
    # Initialize the weighted sum for each layer
    weighted_sum = []
    
    # Ensure that we are aggregating layer by layer
    for layer_weights in zip(*client_weights):  # Zips together the weights for each layer across clients
        # Calculate the weighted sum for this layer
        layer_weight_shape = layer_weights[0].shape  # Get the shape of the weights of the first client
        layer_weight_sum = np.zeros(layer_weight_shape)  # Initialize an array of the same shape as the layer weights
        
        # Calculate weighted sum for this layer based on client data sizes
        total_size = sum(client_data_sizes)  # Total dataset size
        for client_id, weights in enumerate(layer_weights):
            weight_size = client_data_sizes[client_id] / total_size  # Weight per client based on dataset size
            layer_weight_sum += weight_size * weights
        
        weighted_sum.append(layer_weight_sum)
    
    log_update("Aggregation completed.")
    return weighted_sum

def save_model(weights):
    # Save aggregated model weights to a file (for later use)
    with open("aggregated_model_weights.json", "w") as f:
        json.dump(weights, f)

def main():
    # Get the number of clients from the command line
    if len(sys.argv) != 2:
        print("Usage: python simulation.py <number_of_clients>")
        sys.exit(1)
    num_clients = int(sys.argv[1])

    # Load and split dataset
    x_train, y_train, x_test, y_test = load_dataset()
    clients = split_dataset(x_train, y_train, num_clients)
    
    # Calculate dataset sizes for each client
    client_data_sizes = [len(client[0]) for client in clients]

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
    client_accuracies = {}
    for client_id, accuracy in client_accuracies.items():
        log_update(f"Client {client_id}: {accuracy:.2f}")

    # Aggregate client weights
    client_weights = list(client_results.values())
    aggregated_weights = aggregate_weights(client_weights, client_data_sizes)

    # Save the aggregated model
    save_model(aggregated_weights)

    # Evaluate the aggregated model
    model = create_model()
    model.set_weights(aggregated_weights)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    log_update(f"\nAggregated Model Accuracy: {accuracy * 100:.2f}%")

    # Save status updates to a JSON file (for the frontend)
    status_updates = {
        "training_status": "Completed",
        "aggregated_accuracy": accuracy
    }
    with open("status_updates.json", "w") as f:
        json.dump(status_updates, f)

    # Save client accuracies to a JSON file
    with open("client_accuracies.json", "w") as f:
        json.dump(client_accuracies, f)

if __name__ == "__main__":
    main()
