# server.py
import os
import base64
import pickle
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import mediapipe as mp

# ---------- Load TFLite model ----------
MODEL_PATH = "hand_gesture_model.tflite"
LABEL_ENCODER_PATH = "label_encoder.pkl"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------- Load label encoder (if exists) ----------
label_encoder = None
try:
    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
except Exception as e:
    print(f"Warning: could not load label_encoder.pkl: {e}")
    label_encoder = None

# Optional mapping in case label_encoder stores numeric-string classes
labels_dict = {'0': 'Dot', '1': 'Dash', '2': 'BlankSpace', '3': 'BackSpace', '4': 'Next'}

# ---------- MediaPipe hands setup ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# ---------- Flask app ----------
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "GestureSpeak server running ✅"})

def extract_42_features_from_image(img_bgr):
    """
    Run MediaPipe on BGR image and return list of 42 floats (x-min(x_), y-min(y_) pairs)
    for the first detected hand. Returns None if no hand or feature length mismatch.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None

    # extract first hand only (you can change to handle multiple)
    hand_landmarks = results.multi_hand_landmarks[0]

    x_ = []
    y_ = []
    for lm in hand_landmarks.landmark:
        x_.append(lm.x)
        y_.append(lm.y)

    data_aux = []
    min_x = min(x_)
    min_y = min(y_)
    for lm in hand_landmarks.landmark:
        data_aux.append(lm.x - min_x)
        data_aux.append(lm.y - min_y)

    if len(data_aux) != 42:
        return None

    return data_aux

@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({"error": "Empty request body"}), 400

        # Accept either "image" (base64 image) or "landmarks" (array of 42 floats)
        if "landmarks" in payload:
            landmarks = payload["landmarks"]
            if not isinstance(landmarks, list) or len(landmarks) != 42:
                return jsonify({"error": "landmarks must be a list of 42 floats"}), 400
            features = np.array([landmarks], dtype=np.float32)

        elif "image" in payload:
            b64 = payload["image"]
            try:
                img_bytes = base64.b64decode(b64)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    return jsonify({"error": "Could not decode image"}), 400
            except Exception as e:
                return jsonify({"error": f"Bad image data: {e}"}), 400

            features_list = extract_42_features_from_image(frame)
            if features_list is None:
                return jsonify({"prediction": "NoHand", "confidence": 0.0})
            features = np.array([features_list], dtype=np.float32)
        else:
            return jsonify({"error": "Expected 'image' (base64) or 'landmarks' (42 floats)"}), 400

        # Ensure shape matches model input
        expected_shape = input_details[0]['shape']
        # many TFLite models expect (1,42)
        # If model expects different shape, you may need to reshape accordingly
        interpreter.set_tensor(input_details[0]['index'], features)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        pred_idx = int(np.argmax(output))
        confidence = float(np.max(output))

        # Map predicted index -> class label using label_encoder if available
        predicted_class = None
        if label_encoder is not None:
            try:
                predicted_class = label_encoder.classes_[pred_idx]
            except Exception:
                predicted_class = str(pred_idx)
        else:
            predicted_class = str(pred_idx)

        # If label_encoder gave numeric string like '0', map to human label
        human_label = labels_dict.get(predicted_class, predicted_class)

        return jsonify({
            "prediction": human_label,
            "raw_prediction": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # debug False in production; set True while testing
    app.run(host="0.0.0.0", port=port, debug=True)
