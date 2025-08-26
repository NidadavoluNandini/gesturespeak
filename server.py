import os
import base64
import time
import threading
import pickle
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import mediapipe as mp

# ----------------------------
# Configuration
# ----------------------------
MODEL_PATH = "morse_model.tflite"
LABEL_ENCODER_PATH = "label_encoder.pkl"
CONFIDENCE_THRESHOLD = 0.85
MIN_STABLE_FRAMES = 8         # number of consecutive frames required to consider a gesture "stable"
STATE_TTL_SECONDS = 30       # remove client states inactive for this many seconds

# Labels fallback (in case label encoder contains numeric strings)
labels_dict = {'0': 'Dot', '1': 'Dash', '2': 'BlankSpace', '3': 'BackSpace', '4': 'Next'}
display_map = {'Dot': '.', 'Dash': '-', 'BlankSpace': ' '}  # used when adding to displayed_text

# ----------------------------
# Load TFLite model
# ----------------------------
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Lock to protect TFLite interpreter & media pipe in multithreaded server
model_lock = threading.Lock()

# ----------------------------
# Load label encoder (if exists)
# ----------------------------
label_encoder = None
try:
    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
except Exception as e:
    print(f"Warning: could not load label_encoder.pkl: {e}")
    label_encoder = None

# ----------------------------
# MediaPipe setup (single instance)
# ----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# ----------------------------
# Per-client state (in-memory)
# ----------------------------
# Keyed by client id (use client IP + optional provided client_id)
client_states = {}
client_states_lock = threading.Lock()

def cleanup_states():
    """Periodically remove stale client states."""
    while True:
        time.sleep(STATE_TTL_SECONDS)
        now = time.time()
        with client_states_lock:
            stale = [k for k, v in client_states.items() if now - v.get("last_seen", 0) > STATE_TTL_SECONDS]
            for k in stale:
                del client_states[k]

cleanup_thread = threading.Thread(target=cleanup_states, daemon=True)
cleanup_thread.start()

def get_client_key(req):
    """Construct a client key. If client sends 'client_id' in JSON, prefer that, else use remote_addr."""
    try:
        payload = req.get_json(silent=True) or {}
        client_id = payload.get("client_id")
    except Exception:
        client_id = None
    if client_id:
        return f"client::{client_id}"
    # fallback to remote addr
    return f"ip::{req.remote_addr}"

# ----------------------------
# Helpers: image -> 42 features
# ----------------------------
def extract_42_features_from_image(img_bgr):
    """
    Run MediaPipe on BGR image and return list of 42 floats (x-min(x_), y-min(y_) pairs)
    for the first detected hand. Returns None if no hand or mismatch.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0]
    x_ = [lm.x for lm in hand_landmarks.landmark]
    y_ = [lm.y for lm in hand_landmarks.landmark]

    min_x = min(x_)
    min_y = min(y_)
    data_aux = []
    for lm in hand_landmarks.landmark:
        data_aux.append(lm.x - min_x)
        data_aux.append(lm.y - min_y)

    if len(data_aux) != 42:
        return None
    return data_aux

# ----------------------------
# Gesture classification wrapper
# ----------------------------
def classify_features(features_np):
    """
    features_np: numpy array shaped (1,42) dtype float32
    returns: (predicted_label_string, confidence_float, raw_pred_index)
    """
    with model_lock:
        interpreter.set_tensor(input_details[0]['index'], features_np)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
    pred_idx = int(np.argmax(output))
    confidence = float(np.max(output))

    # map index to label using label_encoder if available
    if label_encoder is not None:
        try:
            raw_label = label_encoder.classes_[pred_idx]
        except Exception:
            raw_label = str(pred_idx)
    else:
        raw_label = str(pred_idx)

    # map numeric-string raw_label to human label if present
    human_label = labels_dict.get(raw_label, raw_label)
    return human_label, confidence, raw_label

# ----------------------------
# State update logic (mirrors your local script)
# ----------------------------
def update_client_state(client_key, predicted_character, confidence):
    """
    Update per-client stability and displayed_text state.
    Returns a dict with fields to return to client.
    """
    with client_states_lock:
        state = client_states.get(client_key)
        if state is None:
            # initialize
            state = {
                "displayed_text": "",
                "current_character": "",
                "last_gesture": "",
                "gesture_stable_count": 0,
                "next_detected": False,
                "last_seen": time.time()
            }
            client_states[client_key] = state

        state["last_seen"] = time.time()

    # If prediction confidence is low, reset stability
    if confidence < CONFIDENCE_THRESHOLD:
        state["gesture_stable_count"] = 0
        state["last_gesture"] = ""
        state["current_character"] = ""
        # no state change
        return {
            "action": "unstable",
            "prediction": "Unknown",
            "confidence": confidence,
            "displayed_text": state["displayed_text"],
            "current_character": state["current_character"],
            "gesture_stable_count": state["gesture_stable_count"]
        }

    # If predicted_character repeated, increase stable count
    if predicted_character == state["last_gesture"]:
        state["gesture_stable_count"] += 1
    else:
        state["gesture_stable_count"] = 1
        state["last_gesture"] = predicted_character

    # When stable enough, decide action
    action = "none"
    if state["gesture_stable_count"] >= MIN_STABLE_FRAMES:
        # If gesture is NEXT -> commit current_character (or BackSpace)
        if predicted_character == "Next" and not state["next_detected"]:
            # commit or apply backspace
            if state["current_character"] == "BackSpace":
                if state["displayed_text"]:
                    state["displayed_text"] = state["displayed_text"][:-1]
            elif state["current_character"] and state["current_character"] not in ["Next", "BackSpace"]:
                char_to_add = display_map.get(state["current_character"], "")
                state["displayed_text"] += char_to_add
            state["next_detected"] = True
            action = "commit"
        elif predicted_character != "Next":
            # update current_character (preview)
            state["current_character"] = predicted_character
            state["next_detected"] = False
            action = "preview"

    return {
        "action": action,
        "prediction": predicted_character,
        "confidence": confidence,
        "displayed_text": state["displayed_text"],
        "current_character": state["current_character"],
        "gesture_stable_count": state["gesture_stable_count"]
    }

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "GestureSpeak server running ✅"})

@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Expects JSON with either:
     - {"image": "<base64 JPEG/PNG>", "client_id": "<optional id>"} OR
     - {"landmarks": [42 floats], "client_id": "<optional id>"}
    Returns JSON:
     {
       "prediction": "Dot"|"Dash"|"Next"|"BackSpace"|"Unknown",
       "confidence": 0.92,
       "displayed_text": "...",
       "current_character": "...",
       "action": "preview" | "commit" | "unstable" | "none",
       "gesture_stable_count": n
     }
    """
    try:
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({"error": "Empty request body"}), 400

        client_key = get_client_key(request)

        # If landmarks provided by client
        if "landmarks" in payload:
            landmarks = payload["landmarks"]
            if not isinstance(landmarks, list) or len(landmarks) != 42:
                return jsonify({"error": "landmarks must be a list of 42 floats"}), 400
            features = np.array([landmarks], dtype=np.float32)
        elif "image" in payload:
            # decode image bytes and run mediapipe -> features
            try:
                img_bytes = base64.b64decode(payload["image"])
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    return jsonify({"error": "Could not decode image"}), 400
            except Exception as e:
                return jsonify({"error": f"Bad image data: {e}"}), 400

            features_list = extract_42_features_from_image(frame)
            if features_list is None:
                # no hand detected -> reset stability for client
                with client_states_lock:
                    st = client_states.get(client_key)
                    if st:
                        st["gesture_stable_count"] = 0
                        st["last_gesture"] = ""
                        st["current_character"] = ""
                        st["next_detected"] = False
                        st["last_seen"] = time.time()
                return jsonify({
                    "prediction": "NoHand",
                    "confidence": 0.0,
                    "displayed_text": client_states.get(client_key, {}).get("displayed_text", ""),
                    "current_character": client_states.get(client_key, {}).get("current_character", ""),
                    "action": "nohand",
                    "gesture_stable_count": 0
                })
            features = np.array([features_list], dtype=np.float32)
        else:
            return jsonify({"error": "Expected 'image' (base64) or 'landmarks' (42 floats)"}), 400

        # classify
        predicted_label, confidence, raw = classify_features(features)

        # update client state with the predicted_label
        result = update_client_state(client_key, predicted_label, confidence)
        # include raw label for debugging
        result["raw_prediction"] = raw

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
