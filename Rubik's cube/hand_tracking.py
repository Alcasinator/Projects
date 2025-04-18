import cv2
import mediapipe as mp

# Initialize MediaPipe hands and drawing tools
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up the webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip and convert the frame
        frame = cv2.flip(frame, 1)  # mirror the image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and detect hands
        results = hands.process(rgb)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the frame
        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # Press Esc to quit
            break

cap.release()
cv2.destroyAllWindows()
