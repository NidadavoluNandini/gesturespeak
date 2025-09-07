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

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="morse_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("labels_encoder.txt", "r") as f:
    labels = [line.strip() for line in f if line.strip()]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

def process_frame(base64_str):
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, "Unknown", 0.0

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    label = "Unknown"
    confidence = 0.0

    if results.multi_hand_landmarks:
        # Extract landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        x_ = [lm.x for lm in hand_landmarks.landmark]
        y_ = [lm.y for lm in hand_landmarks.landmark]

        data_aux = []
        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

        if len(data_aux) == 42:
            input_data = np.array([data_aux], dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]

            max_idx = int(np.argmax(output_data))
            confidence = float(output_data[max_idx])
            label = labels[max_idx] if max_idx < len(labels) else "Unknown"

        # Draw hand + label on frame
        for lm in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, lm, mp.solutions.hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"{label} ({confidence:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2, cv2.LINE_AA)

    # Encode modified frame as base64
    _, buffer = cv2.imencode(".jpg", frame)
    frame_b64 = base64.b64encode(buffer).decode("utf-8")

    return frame_b64, label, confidence

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        image_b64 = data.get("image")
        if not image_b64:
            return jsonify({"error": "Missing image"}), 400

        processed_frame, label, confidence = process_frame(image_b64)

        return jsonify({
            "label": label,
            "confidence": confidence,
            "frame": processed_frame
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": "Server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
