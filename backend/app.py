from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import threading
import time
import json
import numpy as np

# Federated learning simulation code
# (You can integrate your existing code here, adapting it to communicate with the frontend)

app = Flask(__name__)
socketio = SocketIO(app)

# Global variables
status_updates = []  # List to store simulation updates
client_accuracies = {}

# Log updates to send to frontend
def log_update(message):
    status_updates.append(message)
    socketio.emit('status_update', message)  # Emit real-time updates to frontend
    print(message)

# Simulate the federated learning process with dummy updates (adapt to your logic)
def run_simulation():
    for i in range(10):  # 10 simulation steps for example
        log_update(f"Simulation step {i+1}")
        time.sleep(1)  # Simulate processing time
    log_update("Simulation completed.")

# Start the simulation in a separate thread
def start_simulation():
    threading.Thread(target=run_simulation).start()

# Route to serve the UI
@app.route('/')
def index():
    return render_template('index.html')  # Serve the frontend

# WebSocket event to control the simulation (e.g., start it)
@socketio.on('start_simulation')
def handle_start_simulation(message):
    log_update("Starting simulation...")
    start_simulation()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
