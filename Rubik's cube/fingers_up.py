import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Finger landmark indices
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                landmarks = hand.landmark
                fingers = []

                # Thumb: Check x instead of y (it's horizontal)
                if landmarks[FINGER_TIPS[0]].x < landmarks[FINGER_PIPS[0]].x:
                    fingers.append(1)  # Up
                else:
                    fingers.append(0)

                # Other 4 fingers
                for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
                    if landmarks[tip].y < landmarks[pip].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Show result on screen
                cv2.putText(frame, f'Fingers Up: {fingers}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Finger Counter", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
