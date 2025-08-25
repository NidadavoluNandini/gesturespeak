import pickle
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

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

print("TensorFlow Lite model loaded successfully!")
print("Press 'q' to quit")

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
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
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

        # Only make prediction if we have the expected number of features (42)
        if len(data_aux) == 42:
            # Prepare input for TensorFlow Lite model
            input_data = np.array([data_aux], dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            interpreter.invoke()
            
            # Get prediction
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_class_idx = np.argmax(output_data[0])
            confidence = np.max(output_data[0])
            
            # Convert class index back to original label
            predicted_class = label_encoder.classes_[predicted_class_idx]
            predicted_character = labels_dict[predicted_class]

            # Display prediction with confidence
            text = f"{predicted_character} ({confidence:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3,
                        cv2.LINE_AA)
        else:
            # Draw bounding box but no prediction for inconsistent data
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(frame, "Multiple hands", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3,
                        cv2.LINE_AA)

    cv2.imshow('TensorFlow Lite Hand Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("TensorFlow Lite inference stopped.")
