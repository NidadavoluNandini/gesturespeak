import os
import base64
import pickle
import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# ----------------------------
# Load TensorFlow Lite Model
# ----------------------------
MODEL_PATH = "morse_model.tflite"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found in working dir")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------------------------
# Load Label Encoder
# ----------------------------
if not os.path.exists("label_encoder.pkl"):
    raise FileNotFoundError("label_encoder.pkl not found in working dir")

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ----------------------------
# Map class → friendly gesture names
# (based on your inference_tflite.py logic)
# ----------------------------
labels_dict = {
    '0': 'Dot',
    '1': 'Dash',
    '2': 'BlankSpace',
    '3': 'BackSpace',
    '4': 'Next'
}

# ----------------------------
# MediaPipe setup
# ----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

# ----------------------------
# Flask + SocketIO setup
# ----------------------------
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

@app.route("/")
def index():
    return "GestureSpeak server running with SocketIO"

# ---- SocketIO Events ----
@socketio.on("connect")
def on_connect():
    print("Client connected")
    emit("server_ready", {"msg": "ok"})

@socketio.on("disconnect")
def on_disconnect():
    print("Client disconnected")

@socketio.on("frame")
def handle_frame(data):
    """Receive base64 JPEG frame, run inference, emit prediction"""
    try:
        b64 = data.get("image")
        if not b64:
            emit("prediction", {"gesture": "Unknown", "confidence": 0.0})
            return

        # decode base64 -> numpy image
        img_bytes = base64.b64decode(b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            emit("prediction", {"gesture": "Unknown", "confidence": 0.0})
            return

        # Run MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            emit("prediction", {"gesture": "Unknown", "confidence": 0.0})
            return

        # Take first hand
        hand_landmarks = results.multi_hand_landmarks[0]
        x_list = [p.x for p in hand_landmarks.landmark]
        y_list = [p.y for p in hand_landmarks.landmark]

        # Build 42 features (normalized like inference_tflite.py)
        data_aux = []
        min_x = min(x_list)
        min_y = min(y_list)
        for p in hand_landmarks.landmark:
            data_aux.append(float(p.x - min_x))
            data_aux.append(float(p.y - min_y))

        if len(data_aux) != 42:
            emit("prediction", {"gesture": "Unknown", "confidence": 0.0})
            return

        # Run TFLite inference
        input_data = np.array([data_aux], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_idx = int(np.argmax(output_data[0]))
        confidence = float(np.max(output_data[0]))

        predicted_class = str(label_encoder.classes_[predicted_class_idx])
        friendly = labels_dict.get(predicted_class, "Unknown")

        if confidence < 0.85:
            emit("prediction", {"gesture": "Unknown", "confidence": confidence})
            return

        emit("prediction", {"gesture": friendly, "confidence": confidence})

    except Exception as e:
        emit("prediction", {"error": str(e)})

# ---- REST endpoint (optional, 42 floats directly) ----
@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    data = body.get("landmarks")
    if not data or len(data) != 42:
        return jsonify({"error": "Expected 42 landmark values"}), 400

    input_data = np.array([data], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_idx = int(np.argmax(output_data[0]))
    confidence = float(np.max(output_data[0]))

    predicted_class = str(label_encoder.classes_[predicted_class_idx])
    friendly = labels_dict.get(predicted_class, "Unknown")

    if confidence < 0.85:
        return jsonify({"gesture": "Unknown", "confidence": confidence})
    return jsonify({"gesture": friendly, "confidence": confidence})

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port)
