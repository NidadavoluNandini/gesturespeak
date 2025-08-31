import os
import threading
from flask import Flask, jsonify
import inference_tflite  # your inference file

app = Flask(__name__)
is_running = False


@app.route('/start_inference', methods=['POST'])
def start_inference():
    """
    Start inference in a background thread so HTTP responds immediately.
    """
    global is_running
    if is_running:
        return jsonify({"status": "already running"})

    def run_model():
        global is_running
        is_running = True
        try:
            inference_tflite.run_inference()   # call your function
        except Exception as e:
            print("Error while running inference:", e, flush=True)
        finally:
            is_running = False

    threading.Thread(target=run_model, daemon=True).start()
    return jsonify({"status": "inference started"})


@app.route('/stop_inference', methods=['POST'])
def stop_inference():
    """
    Currently, inference_tflite must handle stopping logic.
    For now we just return a message.
    """
    return jsonify({"status": "stop not implemented in server, press 'q' locally"})


@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "✅ GestureSpeak server is running"})


if __name__ == '__main__':
    # Railway provides PORT env var
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
