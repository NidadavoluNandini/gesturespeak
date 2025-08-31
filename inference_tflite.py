import pickle
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import os

# Detect runtime environment
RUN_MODE = os.getenv("RUN_MODE", "server")  # "local" or "server"

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='./morse_model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera with index 0")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open any camera")
        exit()
    else:
        print("Using camera index 1")
else:
    print("Using camera index 0")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {'0': 'Dot', '1': 'Dash', '2': 'BlankSpace', '3': 'BackSpace', '4': 'Next'}
display_map = {'Dot': '.', 'Dash': '-', 'BlankSpace': ' '}

# Text accumulation
displayed_text = ""
current_character = ""
last_gesture = ""
gesture_stable_count = 0
min_stable_frames = 10
next_detected = False

# Clear button properties (only for local mode)
clear_button_width = 80
clear_button_height = 40
clear_button_x = 0
clear_button_y = 90

def mouse_callback(event, x, y, flags, param):
    global displayed_text
    if event == cv2.EVENT_LBUTTONDOWN and clear_button_x <= x <= clear_button_x + clear_button_width and clear_button_y <= y <= clear_button_y + clear_button_height:
        displayed_text = ""
        print("Cleared all text via button click")

print("TensorFlow Lite model loaded successfully!")
print("Press 'q' to quit (local mode)")

# Local debugging only
if RUN_MODE == "local":
    window_name = 'TensorFlow Lite Hand Gesture Recognition'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

def run_inference():
    global displayed_text, current_character, last_gesture, gesture_stable_count, next_detected

    while True:
        data_aux = []
        x_, y_ = [], []

        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to capture frame from camera")
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

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
                                displayed_text += display_map.get(current_character, "")
                            next_detected = True
                        elif predicted_character != "Next":
                            current_character = predicted_character
                            next_detected = False

        # LOCAL ONLY: Show window
        if RUN_MODE == "local":
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # SERVER MODE: Break after 1 cycle (so it returns result)
        if RUN_MODE == "server":
            return displayed_text

    cap.release()
    if RUN_MODE == "local":
        cv2.destroyAllWindows()
    print("TensorFlow Lite inference stopped.")
    return displayed_text
