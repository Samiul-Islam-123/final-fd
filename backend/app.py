from flask import Flask, jsonify, request
import subprocess

app = Flask(__name__)

# Endpoint to start the simulation
@app.route('/start', methods=['POST'])
def start_simulation():
    subprocess.Popen(['python3', 'backend/simulation.py'])
    return jsonify({"message": "Federated Learning simulation started"}), 200

# Endpoint to stop the simulation
@app.route('/stop', methods=['POST'])
def stop_simulation():
    return jsonify({"message": "Federated Learning simulation cannot be stopped manually in this simple setup"}), 200

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"message": "Backend is running"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
