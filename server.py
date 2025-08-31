from flask import Flask, jsonify, request
import threading
import inference  # your existing inference.py file

app = Flask(__name__)

is_running = False

@app.route('/start_inference', methods=['POST'])
def start_inference():
    global is_running
    if is_running:
        return jsonify({"status": "already running"})

    def run_model():
        global is_running
        is_running = True
        try:
            inference.run_inference()   # call function inside inference.py
        except Exception as e:
            print("Error:", e)
        finally:
            is_running = False

    # Run inference in background thread
    threading.Thread(target=run_model).start()

    return jsonify({"status": "inference started"})

@app.route('/stop_inference', methods=['POST'])
def stop_inference():
    # You can add logic to stop your inference loop
    return jsonify({"status": "inference stopped"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
