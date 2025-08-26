import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---------------------------
# Load TFLite model
# ---------------------------
interpreter = tf.lite.Interpreter(model_path="hand_gesture_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels for your model
class_labels = ["Dot", "Dash", "Next", "Del", "Space"]

# ---------------------------
# Flask app setup
# ---------------------------
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "✅ GestureSpeak server running with HTTP API"

# ---------------------------
# Prediction endpoint
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if "landmarks" not in data:
            return jsonify({"error": "No landmarks provided"}), 400

        # Expecting landmarks as a flat list of 42 floats
        landmarks = np.array(data["landmarks"], dtype=np.float32).reshape(1, -1)

        # Run inference
        interpreter.set_tensor(input_details[0]["index"], landmarks)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])

        # Get predicted class
        predicted_idx = int(np.argmax(output_data))
        predicted_class = class_labels[predicted_idx]

        return jsonify({
            "prediction": predicted_class,
            "confidence": float(np.max(output_data))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
