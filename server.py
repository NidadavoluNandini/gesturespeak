from flask import Flask, request, jsonify
import base64
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Health check route for Railway
@app.route("/")
def health():
    return "GestureSpeak server is running"

# Load TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path="morse_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Load labels
try:
    with open("labels_encoder.txt", "r") as f:
        labels = [line.strip() for line in f if line.strip()]
except Exception as e:
    print(f"Error loading labels: {e}")
    labels = []

def extract_landmarks_from_image(base64_str):
    try:
        # Decode base64 image (placeholder logic)
        # Replace with actual landmark extraction later
        _ = base64.b64decode(base64_str)  # Just to validate input
        return np.full((1, 42), 0.5, dtype=np.float32)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        image_b64 = data.get("image")

        if not image_b64:
            return jsonify({"error": "Missing image"}), 400

        input_data = extract_landmarks_from_image(image_b64)
        if input_data is None:
            return jsonify({"error": "Invalid image data"}), 400

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        max_idx = int(np.argmax(output_data))
        confidence = float(output_data[max_idx])
        label = labels[max_idx] if max_idx < len(labels) else "Unknown"

        return jsonify({"label": label, "confidence": confidence})
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": "Server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
