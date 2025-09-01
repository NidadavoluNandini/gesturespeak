import pickle
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='./morse_model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {'0': 'Dot', '1': 'Dash', '2': 'BlankSpace', '3': 'BackSpace', '4': 'Next'}
display_map = {'Dot': '.', 'Dash': '-', 'BlankSpace': ' '}

# Global state
running = False


def run_inference():
    global running
    running = True

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return

    displayed_text = ""
    current_character = ""
    last_gesture = ""
    gesture_stable_count = 0
    min_stable_frames = 10
    next_detected = False

    print("✅ TensorFlow Lite model loaded successfully! Starting inference loop...")

    while running:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("❌ Error: Failed to capture frame")
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        data_aux, x_, y_ = [], [], []

        if not results.multi_hand_landmarks:
            gesture_stable_count = 0
            last_gesture = ""
            current_character = ""

        if results.multi_hand_landmarks:
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
                                char_to_add = display_map.get(current_character, "")
                                displayed_text += char_to_add
                            next_detected = True
                        elif predicted_character != "Next":
                            current_character = predicted_character
                            next_detected = False

                        print(f"✅ Gesture: {predicted_character} | Current Text: {displayed_text}")
                else:
                    gesture_stable_count = 0

        time.sleep(0.05)  # small delay for CPU relief

    cap.release()
    print("🛑 Inference loop stopped.")


def stop_inference():
    global running
    running = False
