from flask import Flask, request, jsonify
import base64
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="morse_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("labels_encoder.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

def extract_landmarks_from_image(base64_str):
    # Decode base64 image (placeholder logic)
    # Replace this with actual landmark extraction (e.g., MediaPipe Hands)
    return np.full((1, 42), 0.5, dtype=np.float32)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_b64 = data.get("image")

    if not image_b64:
        return jsonify({"error": "Missing image"}), 400

    input_data = extract_landmarks_from_image(image_b64)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    max_idx = int(np.argmax(output_data))
    confidence = float(output_data[max_idx])
    label = labels[max_idx]

    return jsonify({"label": label, "confidence": confidence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
