import os, base64, time, threading, pickle
import cv2, numpy as np, tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import mediapipe as mp

# Config
MODEL_PATH = "morse_model.tflite"
LABEL_ENCODER_PATH = "label_encoder.pkl"
CONFIDENCE_THRESHOLD = 0.85

# Labels
labels_dict = {'0': 'Dot', '1': 'Dash', '2': 'BlankSpace', '3': 'BackSpace', '4': 'Next'}
display_map = {'Dot': '.', 'Dash': '-', 'BlankSpace': ' '}

# Load TFLite
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Label encoder
try:
    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
except:
    label_encoder = None

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Flask app
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "GestureSpeak server running ✅"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data or "image" not in data:
            return jsonify({"error": "Expected base64 image"}), 400

        # Decode image
        img_bytes = base64.b64decode(data["image"])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Extract features with MediaPipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            return jsonify({"prediction": "NoHand", "confidence": 0.0})

        hand_landmarks = results.multi_hand_landmarks[0]
        x_ = [lm.x for lm in hand_landmarks.landmark]
        y_ = [lm.y for lm in hand_landmarks.landmark]
        min_x, min_y = min(x_), min(y_)
        features = []
        for lm in hand_landmarks.landmark:
            features.append(lm.x - min_x)
            features.append(lm.y - min_y)

        if len(features) != 42:
            return jsonify({"prediction": "InvalidFeatures"})

        features_np = np.array([features], dtype=np.float32)

        # Run model
        interpreter.set_tensor(input_details[0]['index'], features_np)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        pred_idx = int(np.argmax(output))
        confidence = float(np.max(output))

        # Map label
        if label_encoder:
            raw_label = label_encoder.classes_[pred_idx]
        else:
            raw_label = str(pred_idx)

        prediction = labels_dict.get(raw_label, raw_label)

        return jsonify({
            "prediction": prediction,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
