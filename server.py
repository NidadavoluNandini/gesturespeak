from flask import Flask, request, jsonify
import base64
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
import os

app = Flask(__name__)

@app.route("/")
def health():
    return "GestureSpeak server is running"

# -----------------------------
# Load TFLite model
# -----------------------------
try:
    interpreter = tf.lite.Interpreter(model_path="morse_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ TFLite model loaded")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# -----------------------------
# Load labels
# -----------------------------
try:
    with open("labels_encoder.txt", "r") as f:
        labels = [line.strip() for line in f if line.strip()]
    print(f"✅ Labels loaded: {labels}")
except Exception as e:
    print(f"❌ Error loading labels: {e}")
    labels = []

# -----------------------------
# Mediapipe hands setup
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

def extract_landmarks_from_image(base64_str):
    """Decode base64 image → extract 21 hand landmarks → return np.array shape (1, 42)."""
    try:
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            print("❌ Error: Failed to decode image")
            return None

        # Convert to RGB for Mediapipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            print("⚠️ No hand detected")
            return None

        # Take first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        x_ = [lm.x for lm in hand_landmarks.landmark]
        y_ = [lm.y for lm in hand_landmarks.landmark]

        data_aux = []
        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

        if len(data_aux) == 42:
            return np.array([data_aux], dtype=np.float32)
        else:
            return None
    except Exception as e:
        print(f"❌ Error extracting landmarks: {e}")
        return None

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        image_b64 = data.get("image")

        if not image_b64:
            return jsonify({"error": "Missing image"}), 400

        input_data = extract_landmarks_from_image(image_b64)
        if input_data is None:
            return jsonify({"label": "Unknown", "confidence": 0.0}), 200

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        max_idx = int(np.argmax(output_data))
        confidence = float(output_data[max_idx])
        label = labels[max_idx] if max_idx < len(labels) else "Unknown"

        return jsonify({"label": label, "confidence": confidence})
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": "Server error"}), 500

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Railway will override $PORT automatically
    app.run(host="0.0.0.0", port=port)
