import os
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

# ----------------------------
# Load TensorFlow Lite Model
# ----------------------------
interpreter = tf.lite.Interpreter(model_path="morse_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------------------------
# Load Label Encoder
# ----------------------------
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ----------------------------
# Flask App Setup
# ----------------------------
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return "GestureSpeak server running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("landmarks")  # Flutter should send list of 42 floats
    
    if not data or len(data) != 42:
        return jsonify({"error": "Expected 42 landmark values"}), 400

    # Prepare input
    input_data = np.array([data], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_idx = int(np.argmax(output_data[0]))
    confidence = float(np.max(output_data[0]))

    # Apply confidence threshold
    if confidence < 0.85:
        return jsonify({"gesture": "Unknown", "confidence": confidence})

    predicted_class = label_encoder.classes_[predicted_class_idx]
    return jsonify({"gesture": str(predicted_class), "confidence": confidence})

# ----------------------------
# Run Server
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Railway sets PORT automatically
    app.run(host="0.0.0.0", port=port)
