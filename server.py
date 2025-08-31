from flask import Flask, jsonify
import threading
import inference_tflite  # our file

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
            inference_tflite.run_inference()
        except Exception as e:
            print("Error:", e)
        finally:
            is_running = False

    threading.Thread(target=run_model).start()
    return jsonify({"status": "inference started"})

@app.route('/stop_inference', methods=['POST'])
def stop_inference():
    # Currently you stop by pressing 'q' in OpenCV window
    return jsonify({"status": "press q in window to stop"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
