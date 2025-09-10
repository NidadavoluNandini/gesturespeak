# server.py

import base64
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
import os
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse

app = FastAPI()

# Health check
@app.get("/")
def health():
    return {"status": "GestureSpeak server is running"}

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="morse_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("label_encoder.txt", "r") as f:
    labels = [line.strip() for line in f if line.strip()]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


def process_frame(base64_str):
    """Decode base64 → run MediaPipe → run TFLite → return label & confidence"""
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return "Unknown", 0.0

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    label = "Unknown"
    confidence = 0.0

    if results.multi_hand_landmarks:
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

    return label, confidence


# ✅ HTTP prediction endpoint (Hugging Face friendly)
@app.post("/predict")
async def predict_http(data: dict):
    try:
        image_b64 = data.get("image")
        if not image_b64:
            return JSONResponse({"error": "No image provided"}, status_code=400)

        label, confidence = process_frame(image_b64)
        return {"label": label, "confidence": confidence}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ✅ WebSocket prediction endpoint
@app.websocket("/ws/predict")
async def predict_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            image_b64 = await ws.receive_text()
            label, confidence = process_frame(image_b64)
            await ws.send_json({"label": label, "confidence": confidence})
    except Exception as e:
        await ws.close(code=1011, reason=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)

