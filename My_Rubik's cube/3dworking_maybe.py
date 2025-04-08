import numpy as np
import cv2
import mediapipe as mp
import math
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *

# Initialize Pygame
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
pygame.display.set_caption("3D Rubik's Cube with Gestures")
clock = pygame.time.Clock()

# Define cube colors
colors = {
    'W': (1.0, 1.0, 1.0),  # White
    'Y': (1.0, 1.0, 0.0),  # Yellow
    'R': (1.0, 0.0, 0.0),  # Red
    'O': (1.0, 0.5, 0.0),  # Orange
    'B': (0.0, 0.0, 1.0),  # Blue
    'G': (0.0, 1.0, 0.0),  # Green
}

# Initialize 2D cube state
cube = {
    'U': np.full((3, 3), 'W'),
    'D': np.full((3, 3), 'Y'),
    'F': np.full((3, 3), 'R'),
    'B': np.full((3, 3), 'O'),
    'L': np.full((3, 3), 'B'),
    'R': np.full((3, 3), 'G'),
}

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# OpenGL setup
def init():
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.1, 0.1, 0.1, 1.0)
    glDisable(GL_CULL_FACE)  # Ensure all faces are visible

def draw_cube_body(x, y, z, size):
    half = size / 2
    glColor3f(0.1, 0.1, 0.1)  # Dark gray/black for cube body
    glBegin(GL_QUADS)
    # Front face
    glVertex3f(x - half, y - half, z + half)
    glVertex3f(x + half, y - half, z + half)
    glVertex3f(x + half, y + half, z + half)
    glVertex3f(x - half, y + half, z + half)
    # Back face
    glVertex3f(x - half, y - half, z - half)
    glVertex3f(x + half, y - half, z - half)
    glVertex3f(x + half, y + half, z - half)
    glVertex3f(x - half, y + half, z - half)
    # Top face
    glVertex3f(x - half, y + half, z - half)
    glVertex3f(x + half, y + half, z - half)
    glVertex3f(x + half, y + half, z + half)
    glVertex3f(x - half, y + half, z + half)
    # Bottom face
    glVertex3f(x - half, y - half, z - half)
    glVertex3f(x + half, y - half, z - half)
    glVertex3f(x + half, y - half, z + half)
    glVertex3f(x - half, y - half, z + half)
    # Left face
    glVertex3f(x - half, y - half, z - half)
    glVertex3f(x - half, y - half, z + half)
    glVertex3f(x - half, y + half, z + half)
    glVertex3f(x - half, y + half, z - half)
    # Right face
    glVertex3f(x + half, y - half, z - half)
    glVertex3f(x + half, y - half, z + half)
    glVertex3f(x + half, y + half, z + half)
    glVertex3f(x + half, y + half, z - half)
    glEnd()

def draw_cubelet(x, y, z, size, stickers):
    half = size / 2
    face_directions = {
        'F': ((0, 0, 1),   (0, 0, 0)),     # Front
        'B': ((0, 0, -1),  (0, 180, 0)),   # Back
        'U': ((0, 1, 0),   (-90, 0, 0)),   # Up
        'D': ((0, -1, 0),  (90, 0, 0)),    # Down
        'L': ((-1, 0, 0),  (0, -90, 0)),   # Left
        'R': ((1, 0, 0),   (0, 90, 0)),    # Right
    }

    glPushMatrix()
    glTranslatef(x, y, z)

    # Draw cubelet body
    draw_cube_body(0, 0, 0, size * 0.98)

    # Draw colored stickers
    for face, color_key in stickers.items():
        glPushMatrix()
        normal, rotation = face_directions[face]
        glTranslatef(*[n * half for n in normal])
        glRotatef(rotation[0], 1, 0, 0)
        glRotatef(rotation[1], 0, 1, 0)
        glRotatef(rotation[2], 0, 0, 1)

        glColor3fv(colors[color_key])
        glBegin(GL_QUADS)
        glVertex3f(-half * 0.9, -half * 0.9, 0.05)
        glVertex3f(half * 0.9, -half * 0.9, 0.05)
        glVertex3f(half * 0.9, half * 0.9, 0.05)
        glVertex3f(-half * 0.9, half * 0.9, 0.05)
        glEnd()
        glPopMatrix()

    glPopMatrix()

def draw_full_cube():
    spacing = 1.1
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if x in [0, 2] or y in [0, 2] or z in [0, 2]:  # Only draw outer cubelets
                    stickers = {}
                    if z == 2: stickers['F'] = cube['F'][2-y][x]    # Front (Red)
                    if z == 0: stickers['B'] = cube['B'][2-y][2-x]  # Back (Orange)
                    if y == 2: stickers['U'] = cube['U'][2-z][x]    # Up (White)
                    if y == 0: stickers['D'] = cube['D'][z][x]      # Down (Yellow)
                    if x == 0: stickers['L'] = cube['L'][2-y][2-z]  # Left (Blue)
                    if x == 2: stickers['R'] = cube['R'][2-y][z]    # Right (Green)
                    draw_cubelet((x-1)*spacing, (y-1)*spacing, (z-1)*spacing, 0.9, stickers)

# Reset cube state
def reset_cube():
    global cube
    cube = {
        'U': np.full((3, 3), 'W'),
        'D': np.full((3, 3), 'Y'),
        'F': np.full((3, 3), 'R'),
        'B': np.full((3, 3), 'O'),
        'L': np.full((3, 3), 'B'),
        'R': np.full((3, 3), 'G'),
    }

# 3D Rotation logic with adjacent face updates
def rotate_face_3d(face, direction):
    global cube
    if face == 'U':
        cube['U'] = np.rot90(cube['U'], -1 if direction == 1 else 1)

        front = cube['F'][0].copy()
        right = cube['R'][0].copy()
        back = cube['B'][0].copy()
        left = cube['L'][0].copy()

        if direction == 1:  # Clockwise
            cube['F'][0] = left[::-1]
            cube['R'][0] = front[::-1]
            cube['B'][0] = right[::-1]
            cube['L'][0] = back[::-1]
        else:  # Counterclockwise
            cube['F'][0] = right[::-1]
            cube['R'][0] = back[::-1]
            cube['B'][0] = left[::-1]
            cube['L'][0] = front[::-1]
    elif face == 'D':
        cube['D'] = np.rot90(cube['D'], -1 if direction == 1 else 1)

        front = cube['F'][2].copy()
        right = cube['R'][2].copy()
        back = cube['B'][2].copy()
        left = cube['L'][2].copy()

        if direction == 1:  # Clockwise
            cube['F'][2] = right[::-1]
            cube['R'][2] = back[::-1]
            cube['B'][2] = left[::-1]
            cube['L'][2] = front[::-1]
        else:  # Counterclockwise
            cube['F'][2] = left[::-1]
            cube['R'][2] = front[::-1]
            cube['B'][2] = right[::-1]
            cube['L'][2] = back[::-1]
    elif face == 'F':
        if direction == 1:
            cube['F'] = np.rot90(cube['F'], -1)
            # Update adjacent faces (U, R, D, L)
            up_row = cube['U'][2].copy()
            right_col = cube['R'][:, 0].copy()
            down_row = cube['D'][0].copy()
            left_col = cube['L'][:, 2].copy()
            cube['U'][2] = left_col[::-1]
            cube['R'][:, 0] = up_row
            cube['D'][0] = right_col[::-1]
            cube['L'][:, 2] = down_row
        else:
            cube['F'] = np.rot90(cube['F'], 1)
            up_row = cube['U'][2].copy()
            right_col = cube['R'][:, 0].copy()
            down_row = cube['D'][0].copy()
            left_col = cube['L'][:, 2].copy()
            cube['U'][2] = right_col[::-1]
            cube['R'][:, 0] = down_row
            cube['D'][0] = left_col[::-1]
            cube['L'][:, 2] = up_row
    elif face == 'B':
        if direction == 1:
            cube['B'] = np.rot90(cube['B'], -1)
            # Update adjacent faces (U, R, D, L)
            up_row = cube['U'][0].copy()
            right_col = cube['R'][:, 2].copy()
            down_row = cube['D'][2].copy()
            left_col = cube['L'][:, 0].copy()
            cube['U'][0] = right_col
            cube['R'][:, 2] = down_row[::-1]
            cube['D'][2] = left_col
            cube['L'][:, 0] = up_row[::-1]
        else:
            cube['B'] = np.rot90(cube['B'], 1)
            up_row = cube['U'][0].copy()
            right_col = cube['R'][:, 2].copy()
            down_row = cube['D'][2].copy()
            left_col = cube['L'][:, 0].copy()
            cube['U'][0] = left_col
            cube['R'][:, 2] = up_row[::-1]
            cube['D'][2] = right_col
            cube['L'][:, 0] = down_row[::-1]
    elif face == 'L':
        if direction == 1:
            cube['L'] = np.rot90(cube['L'], -1)
            # Update adjacent faces (U, F, D, B)
            up_col = cube['U'][:, 0].copy()
            front_col = cube['F'][:, 0].copy()
            down_col = cube['D'][:, 0].copy()
            back_col = cube['B'][:, 2].copy()
            cube['U'][:, 0] = back_col
            cube['F'][:, 0] = up_col
            cube['D'][:, 0] = front_col
            cube['B'][:, 2] = down_col[::-1]
        else:
            cube['L'] = np.rot90(cube['L'], 1)
            up_col = cube['U'][:, 0].copy()
            front_col = cube['F'][:, 0].copy()
            down_col = cube['D'][:, 0].copy()
            back_col = cube['B'][:, 2].copy()
            cube['U'][:, 0] = front_col
            cube['F'][:, 0] = down_col
            cube['D'][:, 0] = back_col[::-1]
            cube['B'][:, 2] = up_col
    elif face == 'R':
        if direction == 1:
            cube['R'] = np.rot90(cube['R'], -1)
            # Update adjacent faces (U, F, D, B)
            up_col = cube['U'][:, 2].copy()
            front_col = cube['F'][:, 2].copy()
            down_col = cube['D'][:, 2].copy()
            back_col = cube['B'][:, 0].copy()
            cube['U'][:, 2] = front_col
            cube['F'][:, 2] = down_col
            cube['D'][:, 2] = back_col[::-1]
            cube['B'][:, 0] = up_col[::-1]
        else:
            cube['R'] = np.rot90(cube['R'], 1)
            up_col = cube['U'][:, 2].copy()
            front_col = cube['F'][:, 2].copy()
            down_col = cube['D'][:, 2].copy()
            back_col = cube['B'][:, 0].copy()
            cube['U'][:, 2] = back_col[::-1]
            cube['F'][:, 2] = up_col
            cube['D'][:, 2] = front_col[::-1]
            cube['B'][:, 0] = down_col

def rotate_middle_horizontal(direction):
    global cube
    temp = cube['F'][1].copy()
    if direction == 1:  # Right rotation (F → R → B → L)
        cube['F'][1] = cube['R'][1]
        cube['R'][1] = cube['B'][1]
        cube['B'][1] = cube['L'][1]
        cube['L'][1] = temp
    else:  # Left rotation (F → L → B → R)
        cube['F'][1] = cube['L'][1]
        cube['L'][1] = cube['B'][1]
        cube['B'][1] = cube['R'][1]
        cube['R'][1] = temp

def rotate_middle_vertical(direction):
    global cube
    temp_col = cube['F'][:, 1][::-1].copy()  # Start with Front middle column, reversed
    if direction == 1:  # Clockwise: Front → Down → Back → Up
        cube['F'][:, 1] = cube['D'][:, 1][::-1]  # Down to Front, reversed
        cube['D'][:, 1] = cube['B'][:, 1]        # Back to Down
        cube['B'][:, 1] = cube['U'][:, 1]  # Up to Back, reversed
        cube['U'][:, 1] = temp_col              # Front to Up
    else:  # Counterclockwise: Front → Up → Back → Down
        cube['F'][:, 1] = cube['D'][:, 1][::-1]  # Down to Front, reversed
        cube['D'][:, 1] = cube['B'][:, 1]        # Back to Down
        cube['B'][:, 1] = cube['U'][:, 1]        # Up to Back
        cube['U'][:, 1] = temp_col               # Front to Up

# Process gestures
def process_gesture(gesture_text):
    global last_gesture
    if gesture_text == "Reset Cube":
        reset_cube()
    elif gesture_text == "Rotate Right Face":
        rotate_face_3d("R", 1)
    elif gesture_text == "Rotate Up Face":
        rotate_face_3d("U", 1)
    elif gesture_text == "Rotate Front Face":
        rotate_face_3d("F", 1)
    elif gesture_text == "Rotate Left Face":
        rotate_face_3d("L", 1)
    elif gesture_text == "Rotate Down Face":
        rotate_face_3d("D", 1)
    elif gesture_text == "Rotate Back Face":
        rotate_face_3d("B", 1)
    elif gesture_text == "Middle Row Right":
        rotate_middle_horizontal(1)
    elif gesture_text == "Middle Row Left":
        rotate_middle_horizontal(-1)
    elif gesture_text == "Middle Column Up":
        rotate_middle_vertical(-1)  # Swapped to counterclockwise
    elif gesture_text == "Middle Column Down":
        rotate_middle_vertical(1)   # Swapped to clockwise
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
            return "Middle Column Up"  # Now counterclockwise
        if is_palm_open(right):
            return "Rotate Front Face"

    # Left hand gestures
    if left:
        if is_index_up(left):
            return "Rotate Left Face"
        if is_pinky_up(left):
            return "Rotate Down Face"
        if is_middle_up(left):
            return "Middle Column Down"  # Now clockwise
        if is_palm_open(left):
            return "Rotate Back Face"

    return ""

# Main loop
last_gesture = ""
waiting_for_rest = False
running = True
init()

# Mouse rotation variables
dragging = False
last_mouse_pos = (0, 0)
x_rot, y_rot = 35.0, 45.0  # Initial isometric view
sensitivity = 0.5

while running:
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -10)
    glRotatef(x_rot, 1, 0, 0)
    glRotatef(y_rot, 0, 1, 0)
    draw_full_cube()

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

    # Handle mouse events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                dragging = True
                last_mouse_pos = pygame.mouse.get_pos()
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if dragging:
                x, y = pygame.mouse.get_pos()
                dx = x - last_mouse_pos[0]
                dy = y - last_mouse_pos[1]
                y_rot += dx * sensitivity
                x_rot += dy * sensitivity
                last_mouse_pos = (x, y)

    pygame.display.flip()
    clock.tick(60)

cap.release()
cv2.destroyAllWindows()
pygame.quit()