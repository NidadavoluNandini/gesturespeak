from flask import Flask, jsonify
import threading
import inference_tflite  # our inference script

app = Flask(__name__)
is_running = False
thread = None


@app.route('/start_inference', methods=['POST'])
def start_inference():
    global is_running, thread
    if is_running:
        return jsonify({"status": "already running"})

    def run_model():
        global is_running
        is_running = True
        try:
            inference_tflite.run_inference()
        except Exception as e:
            print("❌ Error in inference:", e)
        finally:
            is_running = False

    thread = threading.Thread(target=run_model, daemon=True)
    thread.start()

    return jsonify({"status": "inference started"})


@app.route('/stop_inference', methods=['POST'])
def stop_inference():
    global is_running
    if not is_running:
        return jsonify({"status": "not running"})

    inference_tflite.stop_inference()
    return jsonify({"status": "stopping inference..."})


@app.route("/", methods=["GET"])
def home():
    return "✅ GestureSpeak server (Python 3.10) is running!"


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
