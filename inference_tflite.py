# inference_tflite.py
import pickle
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

def run_inference():
    # Load the TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path='./morse_model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load the label encoder
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    labels_dict = {'0': 'Dot', '1': 'Dash', '2': 'BlankSpace', '3': 'BackSpace', '4': 'Next'}

    print("✅ Model loaded. Press 'q' to quit.")

    while True:
        data_aux, x_, y_ = [], [], []
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

            if len(data_aux) == 42:
                input_data = np.array([data_aux], dtype=np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                idx = np.argmax(output_data[0])
                confidence = np.max(output_data[0])

                predicted_class = label_encoder.classes_[idx]
                predicted_character = labels_dict[predicted_class]

                cv2.putText(frame,
                            f"{predicted_character} ({confidence:.2f})",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow('GestureSpeak Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Inference stopped.")
