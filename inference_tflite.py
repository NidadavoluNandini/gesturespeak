import pickle
import mediapipe as mp
import numpy as np
import tensorflow as tf
import cv2
import time

# -------------------------------
# Load TensorFlow Lite model
# -------------------------------
interpreter = tf.lite.Interpreter(model_path='./morse_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------------
# Load label encoder
# -------------------------------
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# -------------------------------
# Mediapipe Setup
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {'0': 'Dot', '1': 'Dash', '2': 'BlankSpace', '3': 'BackSpace', '4': 'Next'}
display_map = {'Dot': '.', 'Dash': '-', 'BlankSpace': ' '}

# -------------------------------
# Global State
# -------------------------------
displayed_text = ""
current_character = ""
last_gesture = ""
gesture_stable_count = 0
min_stable_frames = 10
next_detected = False
running = True  # To allow stopping later


def run_inference():
    """
    Run hand gesture inference loop (headless).
    Works in Railway – no imshow.
    """
    global displayed_text, current_character, last_gesture
    global gesture_stable_count, next_detected, running

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No camera available in this environment")
        return

    print("✅ Inference started (press Ctrl+C to stop in Railway logs)")

    while running:
        data_aux, x_, y_ = [], [], []
        ret, frame = cap.read()

        if not ret or frame is None:
            print("⚠️ No frame from camera, stopping inference")
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            gesture_stable_count = 0
            last_gesture = ""
            current_character = ""
        else:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            if len(data_aux) == 42:
                input_data = np.array([data_aux], dtype=np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                predicted_class_idx = np.argmax(output_data[0])
                confidence = np.max(output_data[0])

                if confidence > 0.85:
                    predicted_class = label_encoder.classes_[predicted_class_idx]
                    predicted_character = labels_dict[predicted_class]

                    if predicted_character == last_gesture:
                        gesture_stable_count += 1
                    else:
                        gesture_stable_count = 0
                        last_gesture = predicted_character

                    if gesture_stable_count >= min_stable_frames:
                        if predicted_character == "Next" and not next_detected:
                            if current_character == "BackSpace":
                                displayed_text = displayed_text[:-1]
                            elif current_character and current_character not in ["Next", "BackSpace"]:
                                displayed_text += display_map.get(current_character, "")
                            next_detected = True
                        elif predicted_character != "Next":
                            current_character = predicted_character
                            next_detected = False

                        print(f"👉 Gesture: {predicted_character}, Text: {displayed_text}")
                else:
                    gesture_stable_count = 0

        time.sleep(0.05)  # Small delay

    cap.release()
    print("🛑 Inference stopped.")


def stop_inference():
    """ Stop the running loop safely """
    global running
    running = False
