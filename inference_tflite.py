import pickle, cv2, mediapipe as mp, numpy as np, tensorflow as tf, time

# MODE: "local" for debugging with cv2 window, "server" for Railway
MODE = "server"

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='./morse_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {'0': 'Dot', '1': 'Dash', '2': 'BlankSpace', '3': 'BackSpace', '4': 'Next'}
display_map = {'Dot': '.', 'Dash': '-', 'BlankSpace': ' '}

# Main function
def run_inference(max_frames=200):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠️ Could not open camera")
        return {"error": "camera not available"}

    displayed_text = ""
    current_character, last_gesture = "", ""
    gesture_stable_count, next_detected = 0, False
    min_stable_frames = 10

    predictions = []  # collect for server mode

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            gesture_stable_count, last_gesture, current_character = 0, "", ""
        else:
            data_aux, x_, y_ = [], [], []
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x, y = hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y
                    x_.append(x); y_.append(y)
                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                    data_aux.append(hand_landmarks.landmark[i].y - min(y_))

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
                        gesture_stable_count, last_gesture = 0, predicted_character

                    if gesture_stable_count >= min_stable_frames:
                        if predicted_character == "Next" and not next_detected:
                            if current_character == "BackSpace":
                                displayed_text = displayed_text[:-1]
                            elif current_character and current_character not in ["Next", "BackSpace"]:
                                displayed_text += display_map.get(current_character, "")
                            next_detected = True
                        elif predicted_character != "Next":
                            current_character, next_detected = predicted_character, False

                        predictions.append(predicted_character)

        # ---- LOCAL MODE (debugging) ----
        if MODE == "local":
            cv2.imshow("Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # ---- SERVER MODE (stop after N frames) ----
        frame_count += 1
        if MODE == "server" and frame_count >= max_frames:
            break

    cap.release()
    if MODE == "local":
        cv2.destroyAllWindows()

    return {"text": displayed_text, "predictions": predictions}
