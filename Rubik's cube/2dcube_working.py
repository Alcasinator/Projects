import pygame
import numpy as np
import cv2
import mediapipe as mp
import math

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((750, 600))
pygame.display.set_caption("Rubik's Cube")
clock = pygame.time.Clock()

# Define cube colors
colors = {
    "W": (255, 255, 255),
    "Y": (255, 255, 0),
    "R": (255, 0, 0),
    "O": (255, 165, 0),
    "B": (0, 0, 255),
    "G": (0, 128, 0),
}

# Initialize cube
cube = {
    "U": np.full((3, 3), "W"),
    "D": np.full((3, 3), "Y"),
    "F": np.full((3, 3), "R"),
    "B": np.full((3, 3), "O"),
    "L": np.full((3, 3), "B"),
    "R": np.full((3, 3), "G"),
}

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Draw face of the cube
def draw_face(face, x_offset, y_offset):
    for i in range(3):
        for j in range(3):
            color = colors[cube[face][i][j]]
            pygame.draw.rect(screen, color, (x_offset + j * 50, y_offset + i * 50, 48, 48))
            pygame.draw.rect(screen, (0, 0, 0), (x_offset + j * 50, y_offset + i * 50, 48, 48), 2)

# Draw full cube layout
def draw_cube():
    screen.fill((30, 30, 30))
    draw_face("U", 200, 50)
    draw_face("L", 50, 200)
    draw_face("F", 200, 200)
    draw_face("R", 350, 200)
    draw_face("B", 500, 200)
    draw_face("D", 200, 350)
    pygame.display.flip()

# Reset cube state
def reset_cube():
    global cube
    cube = {
        "U": np.full((3, 3), "W"),
        "D": np.full((3, 3), "Y"),
        "F": np.full((3, 3), "R"),
        "B": np.full((3, 3), "O"),
        "L": np.full((3, 3), "B"),
        "R": np.full((3, 3), "G"),
    }
    print("Cube reset to initial state.")

# Rotation logic
def rotate_face_clockwise(face):
    global cube
    cube[face] = np.rot90(cube[face], -1)

    if face == "R":
        top_row = cube["U"][:, 2].copy()
        front_col = cube["F"][:, 2].copy()
        bottom_row = cube["D"][:, 2].copy()
        back_col = cube["B"][:, 0].copy()
        cube["U"][:, 2] = front_col
        cube["F"][:, 2] = bottom_row
        cube["D"][:, 2] = back_col[::-1]
        cube["B"][:, 0] = top_row[::-1]
    elif face == "U":
        front_row = cube["F"][0].copy()
        right_row = cube["R"][0].copy()
        back_row = cube["B"][0].copy()
        left_row = cube["L"][0].copy()
        cube["F"][0] = right_row
        cube["R"][0] = back_row
        cube["B"][0] = left_row
        cube["L"][0] = front_row
    elif face == "F":
        top_row = cube["U"][2].copy()
        left_col = cube["L"][:, 2].copy()
        bottom_row = cube["D"][0].copy()
        right_col = cube["R"][:, 0].copy()
        cube["U"][2] = left_col[::-1]
        cube["L"][:, 2] = bottom_row
        cube["D"][0] = right_col[::-1]
        cube["R"][:, 0] = top_row
    elif face == "L":
        top_row = cube["U"][:, 0].copy()
        front_col = cube["F"][:, 0].copy()
        bottom_row = cube["D"][:, 0].copy()
        back_col = cube["B"][:, 2].copy()
        cube["U"][:, 0] = back_col
        cube["F"][:, 0] = top_row
        cube["D"][:, 0] = front_col
        cube["B"][:, 2] = bottom_row[::-1]
    elif face == "D":
        front_row = cube["F"][2].copy()
        right_row = cube["R"][2].copy()
        back_row = cube["B"][2].copy()
        left_row = cube["L"][2].copy()
        cube["F"][2] = left_row
        cube["R"][2] = front_row
        cube["B"][2] = right_row
        cube["L"][2] = back_row
    elif face == "B":
        top_row = cube["U"][0].copy()
        left_col = cube["L"][:, 0].copy()
        bottom_row = cube["D"][2].copy()
        right_col = cube["R"][:, 2].copy()
        cube["U"][0] = right_col
        cube["L"][:, 0] = top_row[::-1]
        cube["D"][2] = left_col
        cube["R"][:, 2] = bottom_row[::-1]

def rotate_middle_horizontal(direction):
    global cube
    print(f"Rotating middle row {direction} (before): F[1]={cube['F'][1]}, R[1]={cube['R'][1]}, B[1]={cube['B'][1]}, L[1]={cube['L'][1]}")
    temp = cube["F"][1].copy()  # Store the initial F[1] value
    if direction == 1:  # Right rotation (F → R → B → L)
        cube["F"][1] = cube["R"][1]
        cube["R"][1] = cube["B"][1]
        cube["B"][1] = cube["L"][1]
        cube["L"][1] = temp
    else:  # direction == -1, Left rotation (F → L → B → R)
        cube["F"][1] = cube["L"][1]
        cube["L"][1] = cube["B"][1]
        cube["B"][1] = cube["R"][1]
        cube["R"][1] = temp
    print(f"Rotating middle row {direction} (after): F[1]={cube['F'][1]}, R[1]={cube['R'][1]}, B[1]={cube['B'][1]}, L[1]={cube['L'][1]}")

def rotate_middle_vertical(direction):
    global cube
    print(f"Rotating middle column {direction} (before): U[:,1]={cube['U'][:, 1]}, L[:,1]={cube['L'][:, 1]}, D[:,1]={cube['D'][:, 1]}, R[:,1]={cube['R'][:, 1]}")
    temp_col = cube["U"][:, 1].copy()
    if direction == 1:
        cube["U"][:, 1] = cube["L"][:, 1]
        cube["L"][:, 1] = cube["D"][:, 1]
        cube["D"][:, 1] = cube["R"][:, 1]
        cube["R"][:, 1] = temp_col
    else:  # direction == -1 (down)
        cube["U"][:, 1] = cube["R"][:, 1]
        cube["R"][:, 1] = cube["D"][:, 1]
        cube["D"][:, 1] = cube["L"][:, 1]
        cube["L"][:, 1] = temp_col
    print(f"Rotating middle column {direction} (after): U[:,1]={cube['U'][:, 1]}, L[:,1]={cube['L'][:, 1]}, D[:,1]={cube['D'][:, 1]}, R[:,1]={cube['R'][:, 1]}")

# Process gestures
def process_gesture(gesture_text):
    global last_gesture
    print(f"Processing gesture: {gesture_text}")
    if gesture_text == "Reset Cube":
        reset_cube()
    elif gesture_text == "Rotate Right Face":
        rotate_face_clockwise("R")
    elif gesture_text == "Rotate Up Face":
        rotate_face_clockwise("U")
    elif gesture_text == "Rotate Front Face":
        rotate_face_clockwise("F")
    elif gesture_text == "Rotate Left Face":
        rotate_face_clockwise("L")
    elif gesture_text == "Rotate Down Face":
        rotate_face_clockwise("D")
    elif gesture_text == "Rotate Back Face":
        rotate_face_clockwise("B")
    elif gesture_text == "Middle Row Right":
        rotate_middle_horizontal(1)
    elif gesture_text == "Middle Row Left":
        rotate_middle_horizontal(-1)
    elif gesture_text == "Middle Column Up":
        rotate_middle_vertical(1)
    elif gesture_text == "Middle Column Down":
        rotate_middle_vertical(-1)
    last_gesture = gesture_text

# Detect gesture from hand landmarks
def detect_gesture(results):
    if not results.multi_hand_landmarks:
        return ""

    hands_detected = []
    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        handedness = results.multi_handedness[idx].classification[0].label
        hands_detected.append({"handedness": handedness, "landmarks": hand_landmarks})

    left = right = None
    for h in hands_detected:
        if h["handedness"] == "Left":
            left = h["landmarks"]
        elif h["handedness"] == "Right":
            right = h["landmarks"]

    def is_index_up(hand):
        return hand.landmark[8].y < hand.landmark[6].y  # Index tip above PIP

    def is_pinky_up(hand):
        return hand.landmark[20].y < hand.landmark[18].y  # Pinky tip above PIP

    def is_middle_up(hand):
        return hand.landmark[12].y < hand.landmark[10].y  # Middle tip above PIP

    def is_fist(hand):
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        fingers_curled = all(hand.landmark[tip].y > hand.landmark[pip].y + 0.02 for tip, pip in zip(finger_tips, finger_pips))
        thumb_tip = hand.landmark[4]
        wrist = hand.landmark[0]
        thumb_tucked = abs(thumb_tip.x - wrist.x) < 0.1
        return fingers_curled and thumb_tucked

    def is_palm_open(hand):
        wrist = hand.landmark[0]
        index = hand.landmark[8]
        middle = hand.landmark[12]
        ring = hand.landmark[16]
        pinky = hand.landmark[20]
        if not all(hand.landmark[i].y < wrist.y for i in [8, 12, 16, 20]):
            return False
        if abs(index.x - pinky.x) < 0.1:
            return False
        palm_vector = np.array([middle.x - wrist.x, middle.y - wrist.y, middle.z - wrist.z])
        index_vector = np.array([index.x - wrist.x, index.y - wrist.y, index.z - wrist.z])
        dot = np.dot(palm_vector, index_vector)
        norm_product = np.linalg.norm(palm_vector) * np.linalg.norm(index_vector)
        if norm_product == 0:
            return False
        angle = math.degrees(math.acos(np.clip(dot / norm_product, -1.0, 1.0)))
        return angle < 45

    # Debug hand states
    if left:
        print(f"Left hand: index_up={is_index_up(left)}, pinky_up={is_pinky_up(left)}, middle_up={is_middle_up(left)}, fist={is_fist(left)}, palm_open={is_palm_open(left)}")
    if right:
        print(f"Right hand: index_up={is_index_up(right)}, pinky_up={is_pinky_up(right)}, middle_up={is_middle_up(right)}, fist={is_fist(right)}, palm_open={is_palm_open(right)}")

    # Both hands gestures
    if left and right:
        if is_palm_open(left) and is_palm_open(right):
            return "Reset Cube"
        if is_index_up(right) and is_fist(left):
            return "Middle Row Right"
        if is_index_up(left) and is_fist(right):
            return "Middle Row Left"

    # Right hand gestures
    if right:
        if is_index_up(right):
            return "Rotate Right Face"
        if is_pinky_up(right):
            return "Rotate Up Face"
        if is_middle_up(right):
            return "Middle Column Up"
        if is_palm_open(right):
            return "Rotate Front Face"

    # Left hand gestures
    if left:
        if is_index_up(left):
            return "Rotate Left Face"
        if is_pinky_up(left):
            return "Rotate Down Face"
        if is_middle_up(left):
            return "Middle Column Down"
        if is_palm_open(left):
            return "Rotate Back Face"

    return ""

# Main loop
last_gesture = ""
waiting_for_rest = False
running = True

while running:
    draw_cube()
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gesture_text = detect_gesture(results)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Handle gestures
    if waiting_for_rest:
        if gesture_text == "":
            waiting_for_rest = False
            last_gesture = ""
    elif gesture_text and gesture_text != last_gesture:
        process_gesture(gesture_text)
        waiting_for_rest = True

    cv2.putText(frame, f"Gesture: {gesture_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Hand Tracking", frame)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    clock.tick(30)

cap.release()
cv2.destroyAllWindows()
pygame.quit()