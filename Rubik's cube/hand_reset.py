import pygame
import numpy as np
import cv2
import mediapipe as mp

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((750, 600))
pygame.display.set_caption("Rubik's Cube")
clock = pygame.time.Clock()

# Define cube colors
colors = {
    "W": (255, 255, 255),  # White
    "Y": (255, 255, 0),    # Yellow
    "R": (255, 0, 0),      # Red
    "O": (255, 165, 0),    # Orange
    "B": (0, 0, 255),      # Blue
    "G": (0, 128, 0),      # Green
}

# Initialize cube
def init_cube():
    return {
        "U": np.full((3, 3), "W"),
        "D": np.full((3, 3), "Y"),
        "F": np.full((3, 3), "R"),
        "B": np.full((3, 3), "O"),
        "L": np.full((3, 3), "B"),
        "R": np.full((3, 3), "G"),
    }

cube = init_cube()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def draw_face(face, x_offset, y_offset):
    for i in range(3):
        for j in range(3):
            color = colors[cube[face][i][j]]
            pygame.draw.rect(screen, color, (x_offset + j * 50, y_offset + i * 50, 48, 48))
            pygame.draw.rect(screen, (0, 0, 0), (x_offset + j * 50, y_offset + i * 50, 48, 48), 2)

def draw_cube():
    screen.fill((30, 30, 30))
    draw_face("U", 200, 50)
    draw_face("L", 50, 200)
    draw_face("F", 200, 200)
    draw_face("R", 350, 200)
    draw_face("B", 500, 200)
    draw_face("D", 200, 350)
    pygame.display.flip()

def reset_cube():
    global cube
    cube = init_cube()

def rotate_face_clockwise(face):
    cube[face] = np.rot90(cube[face], -1)
    if face == "R":
        top, front, bottom, back = cube["U"][:, 2].copy(), cube["F"][:, 2].copy(), cube["D"][:, 2].copy(), cube["B"][:, 0].copy()
        cube["U"][:, 2], cube["F"][:, 2], cube["D"][:, 2], cube["B"][:, 0] = front, bottom, back[::-1], top[::-1]
    elif face == "U":
        f, r, b, l = cube["F"][0].copy(), cube["R"][0].copy(), cube["B"][0].copy(), cube["L"][0].copy()
        cube["F"][0], cube["R"][0], cube["B"][0], cube["L"][0] = r, b, l, f
    elif face == "F":
        top, left, bottom, right = cube["U"][2].copy(), cube["L"][:, 2].copy(), cube["D"][0].copy(), cube["R"][:, 0].copy()
        cube["U"][2], cube["L"][:, 2], cube["D"][0], cube["R"][:, 0] = left[::-1], bottom, right[::-1], top
    elif face == "L":
        top, front, bottom, back = cube["U"][:, 0].copy(), cube["F"][:, 0].copy(), cube["D"][:, 0].copy(), cube["B"][:, 2].copy()
        cube["U"][:, 0], cube["F"][:, 0], cube["D"][:, 0], cube["B"][:, 2] = back, top, front, bottom[::-1]
    elif face == "D":
        f, r, b, l = cube["F"][2].copy(), cube["R"][2].copy(), cube["B"][2].copy(), cube["L"][2].copy()
        cube["F"][2], cube["R"][2], cube["B"][2], cube["L"][2] = l, f, r, b
    elif face == "B":
        top, left, bottom, right = cube["U"][0].copy(), cube["L"][:, 0].copy(), cube["D"][2].copy(), cube["R"][:, 2].copy()
        cube["U"][0], cube["L"][:, 0], cube["D"][2], cube["R"][:, 2] = right, top[::-1], left, bottom[::-1]

def rotate_middle_horizontal(direction):
    cube["F"][1], cube["R"][1], cube["B"][1], cube["L"][1] = (
        cube["L"][1], cube["F"][1], cube["R"][1], cube["B"][1]
    ) if direction == 1 else (
        cube["R"][1], cube["B"][1], cube["L"][1], cube["F"][1]
    )

def rotate_middle_vertical(direction):
    col = cube["U"][:, 1].copy()
    if direction == 1:
        cube["U"][:, 1] = cube["L"][:, 1]
        cube["L"][:, 1] = cube["D"][:, 1]
        cube["D"][:, 1] = cube["R"][:, 1]
        cube["R"][:, 1] = col
    else:
        cube["U"][:, 1] = cube["R"][:, 1]
        cube["R"][:, 1] = cube["D"][:, 1]
        cube["D"][:, 1] = cube["L"][:, 1]
        cube["L"][:, 1] = col

def process_gesture(gesture_text):
    global last_gesture
    if gesture_text and gesture_text != last_gesture:
        print(f"Operation: {gesture_text}")
        if gesture_text == "Reset Cube":
            reset_cube()
        elif gesture_text == "Rotate Right Face Clockwise":
            rotate_face_clockwise("R")
        elif gesture_text == "Rotate Up Face Clockwise":
            rotate_face_clockwise("U")
        elif gesture_text == "Rotate Front Face Clockwise":
            rotate_face_clockwise("F")
        elif gesture_text == "Rotate Left Face Clockwise":
            rotate_face_clockwise("L")
        elif gesture_text == "Rotate Down Face Clockwise":
            rotate_face_clockwise("D")
        elif gesture_text == "Rotate Back Face Clockwise":
            rotate_face_clockwise("B")
        elif gesture_text == "Middle Row → Right":
            rotate_middle_horizontal(1)
        elif gesture_text == "Middle Row → Left":
            rotate_middle_horizontal(-1)
        elif gesture_text == "Middle Column ↑":
            rotate_middle_vertical(1)
        elif gesture_text == "Middle Column ↓":
            rotate_middle_vertical(-1)
        last_gesture = gesture_text

def detect_gesture(multi_hand_landmarks, multi_handedness):
    if not multi_hand_landmarks or not multi_handedness:
        return ""

    hands_dict = {"Left": None, "Right": None}
    for i, hand_landmark in enumerate(multi_hand_landmarks):
        label = multi_handedness[i].classification[0].label
        hands_dict[label] = hand_landmark

    def is_index_up(hand):
        return hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y

    def is_index_and_middle_up(hand):
        return (hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
                hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y)

    def is_fist(hand):
        return all(
            hand.landmark[finger].y > hand.landmark[finger - 2].y
            for finger in [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                           mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                           mp_hands.HandLandmark.RING_FINGER_TIP,
                           mp_hands.HandLandmark.PINKY_TIP]
        )

    def is_palm_open(hand):
        spread = abs(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x - hand.landmark[mp_hands.HandLandmark.PINKY_TIP].x) > 0.1
        up = all(hand.landmark[f].y < hand.landmark[mp_hands.HandLandmark.WRIST].y for f in [
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP
        ])
        return spread and up

    left, right = hands_dict["Left"], hands_dict["Right"]

    if left and right:
        if is_palm_open(left) and is_palm_open(right):
            return "Reset Cube"
        elif is_index_up(right) and is_fist(left):
            return "Middle Row → Right"
        elif is_index_up(left) and is_fist(right):
            return "Middle Row → Left"
        elif is_index_up(left) and is_index_up(right):
            return "Middle Column ↑"
        elif is_index_and_middle_up(left) and is_index_and_middle_up(right):
            return "Middle Column ↓"

    if right:
        if is_index_up(right): return "Rotate Right Face Clockwise"
        elif is_fist(right): return "Rotate Up Face Clockwise"
        elif is_palm_open(right): return "Rotate Front Face Clockwise"
    if left:
        if is_index_up(left): return "Rotate Left Face Clockwise"
        elif is_fist(left): return "Rotate Down Face Clockwise"
        elif is_palm_open(left): return "Rotate Back Face Clockwise"

    return ""

# Main loop
last_gesture = ""
running = True
while running:
    draw_cube()
    ret, frame = cap.read()
    if not ret:
        print("Warning: Frame capture failed.")
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = ""
    try:
        if results.multi_hand_landmarks:
            gesture = detect_gesture(results.multi_hand_landmarks, results.multi_handedness)
            for hl in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
        else:
            print("No hands detected.")
    except Exception as e:
        print(f"Gesture detection error: {e}")
        continue

    process_gesture(gesture)

    if gesture:
        cv2.putText(frame, f"Operation: {gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Tracking", frame)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    clock.tick(15)

cap.release()
cv2.destroyAllWindows()
pygame.quit()
