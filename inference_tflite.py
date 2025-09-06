import pickle
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='./hand_gesture_model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera with index 0")
    # Try index 1 as backup
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

# New mapping for display purposes
display_map = {'Dot': '.', 'Dash': '-', 'BlankSpace': ' '}

# Text accumulation variables from the example
displayed_text = ""
current_character = ""
last_gesture = ""
gesture_stable_count = 0
min_stable_frames = 10  # Gesture must be stable for this many frames
next_detected = False

# Clear button properties
clear_button_width = 80
clear_button_height = 40
clear_button_x = 0  # Will be set dynamically
clear_button_y = 90

# Mouse callback function for button clicks
def mouse_callback(event, x, y, flags, param):
    global displayed_text
    # Check if click is on Clear button area
    if event == cv2.EVENT_LBUTTONDOWN and clear_button_x <= x <= clear_button_x + clear_button_width and clear_button_y <= y <= clear_button_y + clear_button_height:
        displayed_text = ""
        print("Cleared all text via button click")

print("TensorFlow Lite model loaded successfully!")
print("Press 'q' to quit")

# Set up mouse callback for the window
window_name = 'TensorFlow Lite Hand Gesture Recognition'
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    if not ret or frame is None:
        print("Error: Failed to capture frame from camera")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Reset stability if no hands are detected
    if not results.multi_hand_landmarks:
        gesture_stable_count = 0
        last_gesture = ""
        current_character = ""
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

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

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

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
                            # Use the display mapping to get the character to add
                            char_to_add = display_map.get(current_character, "")
                            displayed_text += char_to_add
                        next_detected = True
                    elif predicted_character != "Next":
                        current_character = predicted_character
                        next_detected = False
                
                box_color = (0, 255, 0) if gesture_stable_count >= min_stable_frames else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
                stability_text = f"Stable: {gesture_stable_count}/{min_stable_frames}"
                cv2.putText(frame, stability_text, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                gesture_stable_count = 0
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(frame, "Multiple hands", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
            gesture_stable_count = 0
            last_gesture = ""
            current_character = ""

    text_bg_height = 80
    cv2.rectangle(frame, (0, 0), (W, text_bg_height), (0, 0, 0), -1)
    cv2.putText(frame, f"Text: {displayed_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    if current_character and current_character not in ["Next"]:
        action_text = " (Show 'Next' to delete)" if current_character == "BackSpace" else " (Show 'Next' to add)"
        cv2.putText(frame, f"Current: {current_character}{action_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    # Draw Clear button
    clear_button_x = W - clear_button_width - 10  # Position on the right side
    cv2.rectangle(frame, (clear_button_x, clear_button_y), (clear_button_x + clear_button_width, clear_button_y + clear_button_height), (0, 0, 255), -1)  # Red button
    cv2.rectangle(frame, (clear_button_x, clear_button_y), (clear_button_x + clear_button_width, clear_button_y + clear_button_height), (255, 255, 255), 2)  # White border
    cv2.putText(frame, "CLEAR", (clear_button_x + 5, clear_button_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("TensorFlow Lite inference stopped.")
