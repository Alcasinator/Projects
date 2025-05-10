import numpy as np
import cv2
import mediapipe as mp
import math
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *
import kociemba

# Initialize Pygame
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
pygame.display.set_caption("3D Rubik's Cube with Gestures and Solver")
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

# Initialize 2D cube state (global)
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

# Solver state (global)
solving = False
solution_steps = []
current_step = 0
expected_face = None
last_gesture = ""

# OpenGL setup
def init():
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.1, 0.1, 0.1, 1.0)
    glDisable(GL_CULL_FACE)

def draw_cube_body(x, y, z, size):
    half = size / 2
    glColor3f(0.1, 0.1, 0.1)
    glBegin(GL_QUADS)
    glVertex3f(x - half, y - half, z + half)
    glVertex3f(x + half, y - half, z + half)
    glVertex3f(x + half, y + half, z + half)
    glVertex3f(x - half, y + half, z + half)
    glVertex3f(x - half, y - half, z - half)
    glVertex3f(x + half, y - half, z - half)
    glVertex3f(x + half, y + half, z - half)
    glVertex3f(x - half, y + half, z - half)
    glVertex3f(x - half, y + half, z - half)
    glVertex3f(x + half, y + half, z - half)
    glVertex3f(x + half, y + half, z + half)
    glVertex3f(x - half, y + half, z + half)
    glVertex3f(x - half, y - half, z - half)
    glVertex3f(x + half, y - half, z - half)
    glVertex3f(x + half, y - half, z + half)
    glVertex3f(x - half, y - half, z + half)
    glVertex3f(x - half, y - half, z - half)
    glVertex3f(x - half, y - half, z + half)
    glVertex3f(x - half, y + half, z + half)
    glVertex3f(x - half, y + half, z - half)
    glVertex3f(x + half, y - half, z - half)
    glVertex3f(x + half, y - half, z + half)
    glVertex3f(x + half, y + half, z + half)
    glVertex3f(x + half, y + half, z - half)
    glEnd()

def draw_cubelet(x, y, z, size, stickers):
    global solving, current_step, solution_steps
    half = size / 2
    face_directions = {
        'F': ((0, 0, 1),   (0, 0, 0)),
        'B': ((0, 0, -1),  (0, 180, 0)),
        'U': ((0, 1, 0),   (-90, 0, 0)),
        'D': ((0, -1, 0),  (90, 0, 0)),
        'L': ((-1, 0, 0),  (0, -90, 0)),
        'R': ((1, 0, 0),   (0, 90, 0)),
    }

    glPushMatrix()
    glTranslatef(x, y, z)
    draw_cube_body(0, 0, 0, size * 0.98)

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

        if solving and current_step < len(solution_steps):
            move = solution_steps[current_step]
            move_face = move[0]
            if len(move) > 1 and move[1] == "'":
                direction = -1
            elif len(move) > 1 and move[1] == "2":
                direction = 2
            else:
                direction = 1
            if move_face == face:
                glColor3f(0.0, 0.0, 0.0)
                draw_arrow(half * 0.9, direction, thickness=0.06)
                glColor3f(1.0, 0.0, 0.0)
                draw_arrow(half * 0.9, direction, thickness=0.04)
        glPopMatrix()
    glPopMatrix()

def draw_arrow(size, direction, thickness):
    glBegin(GL_LINES)
    if direction == 1:
        glVertex3f(size * 0.7, -size * 0.7, 0.06)
        glVertex3f(size * 0.7, size * 0.7, 0.06)
        glVertex3f(size * 0.7, size * 0.7, 0.06)
        glVertex3f(size * 0.6, size * 0.5, 0.06)
        glVertex3f(size * 0.7, size * 0.7, 0.06)
        glVertex3f(size * 0.8, size * 0.5, 0.06)
    elif direction == -1:
        glVertex3f(size * 0.7, size * 0.7, 0.06)
        glVertex3f(size * 0.7, -size * 0.7, 0.06)
        glVertex3f(size * 0.7, -size * 0.7, 0.06)
        glVertex3f(size * 0.6, -size * 0.5, 0.06)
        glVertex3f(size * 0.7, -size * 0.7, 0.06)
        glVertex3f(size * 0.8, -size * 0.5, 0.06)
    elif direction == 2:
        glVertex3f(size * 0.7, -size * 0.7, 0.06)
        glVertex3f(size * 0.7, size * 0.7, 0.06)
        glVertex3f(size * 0.7, size * 0.7, 0.06)
        glVertex3f(size * 0.6, size * 0.5, 0.06)
        glVertex3f(size * 0.7, size * 0.7, 0.06)
        glVertex3f(size * 0.8, size * 0.5, 0.06)
        glVertex3f(size * 0.7, -size * 0.7, 0.06)
        glVertex3f(size * 0.6, -size * 0.5, 0.06)
        glVertex3f(size * 0.7, -size * 0.7, 0.06)
        glVertex3f(size * 0.8, -size * 0.5, 0.06)
    glEnd()

def draw_full_cube():
    spacing = 1.1
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if x in [0, 2] or y in [0, 2] or z in [0, 2]:
                    stickers = {}
                    if z == 2: stickers['F'] = cube['F'][2-y][x]
                    if z == 0: stickers['B'] = cube['B'][2-y][2-x]
                    if y == 2: stickers['U'] = cube['U'][2-z][x]
                    if y == 0: stickers['D'] = cube['D'][z][x]
                    if x == 0: stickers['L'] = cube['L'][2-y][2-z]
                    if x == 2: stickers['R'] = cube['R'][2-y][z]
                    draw_cubelet((x-1)*spacing, (y-1)*spacing, (z-1)*spacing, 0.9, stickers)

def draw_button():
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, display[0], 0, display[1], -1, 1)  # Bottom-left origin, Y up
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    button_x, button_y = 50, 50  # Bottom-left
    button_width, button_height = 100, 50
    glColor3f(0.0, 1.0, 0.0) if not solving else glColor3f(0.5, 0.5, 0.5)
    glBegin(GL_QUADS)
    glVertex2f(button_x, button_y)                  # (50, 50)
    glVertex2f(button_x + button_width, button_y)   # (150, 50)
    glVertex3f(button_x + button_width, button_y + button_height, 0)  # (150, 100)
    glVertex3f(button_x, button_y + button_height, 0)                 # (50, 100)
    glEnd()

    # Text rendering
    font = pygame.font.SysFont("arial", 24)
    text = font.render("Solve", True, (255, 255, 255))
    text_surface = pygame.image.tostring(text, "RGBA", True)
    text_width, text_height = text.get_size()
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_width, text_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_surface)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    text_x = button_x + (button_width - text_width) / 2
    text_y = button_y + (button_height - text_height) / 2
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1); glVertex2f(text_x, text_y)
    glTexCoord2f(1, 1); glVertex2f(text_x + text_width, text_y)
    glTexCoord2f(1, 0); glVertex2f(text_x + text_width, text_y + text_height)
    glTexCoord2f(0, 0); glVertex2f(text_x, text_y + text_height)
    glEnd()
    glDeleteTextures(1, [tex_id])
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    return button_x, button_y, button_width, button_height

def reset_cube():
    global cube, solving, solution_steps, current_step, expected_face
    cube = {
        'U': np.full((3, 3), 'W'),
        'D': np.full((3, 3), 'Y'),
        'F': np.full((3, 3), 'R'),
        'B': np.full((3, 3), 'O'),
        'L': np.full((3, 3), 'B'),
        'R': np.full((3, 3), 'G'),
    }
    solving = False
    solution_steps = []
    current_step = 0
    expected_face = None

def rotate_face_3d(face, direction):
    global cube
    if face == 'U':
        cube['U'] = np.rot90(cube['U'], 1 if direction == 1 else -1)
        front = cube['F'][0].copy()
        right = cube['R'][0].copy()
        back = cube['B'][0].copy()
        left = cube['L'][0].copy()
        if direction == 1:
            cube['F'][0] = right[::-1]
            cube['R'][0] = back[::-1]
            cube['B'][0] = left[::-1]
            cube['L'][0] = front[::-1]            
        else:
            cube['F'][0] = left[::-1]
            cube['R'][0] = front[::-1]
            cube['B'][0] = right[::-1]
            cube['L'][0] = back[::-1]
            
    elif face == 'D':
        cube['D'] = np.rot90(cube['D'], 1 if direction == 1 else -1)
        front = cube['F'][2].copy()
        right = cube['R'][2].copy()
        back = cube['B'][2].copy()
        left = cube['L'][2].copy()
        if direction == 1:
            cube['F'][2] = left[::-1]
            cube['R'][2] = front[::-1]
            cube['B'][2] = right[::-1]
            cube['L'][2] = back[::-1]            
        else:
            cube['F'][2] = right[::-1]
            cube['R'][2] = back[::-1]
            cube['B'][2] = left[::-1]
            cube['L'][2] = front[::-1]
    elif face == 'F':
        if direction == 1:
            cube['F'] = np.rot90(cube['F'], -1)
            up_row = cube['U'][0].copy()
            right_col = cube['R'][:, 2].copy()
            down_row = cube['D'][2].copy()
            left_col = cube['L'][:, 0].copy()
            cube['U'][0] = left_col[::-1]
            cube['R'][:, 2] = up_row
            cube['D'][2] = right_col[::-1]
            cube['L'][:, 0] = down_row
        else:
            cube['F'] = np.rot90(cube['F'], 1)
            up_row = cube['U'][0].copy()
            right_col = cube['R'][:, 2].copy()
            down_row = cube['D'][2].copy()
            left_col = cube['L'][:, 0].copy()
            cube['U'][0] = right_col
            cube['R'][:, 2] = down_row[::-1]
            cube['D'][2] = left_col
            cube['L'][:, 0] = up_row[::-1]
    elif face == 'B':
        if direction == 1:
            cube['B'] = np.rot90(cube['B'], -1)
            up_row = cube['U'][2].copy()
            right_col = cube['R'][:, 0].copy()
            down_row = cube['D'][0].copy()
            left_col = cube['L'][:, 2].copy()
            cube['U'][2] = right_col
            cube['R'][:, 0] = down_row[::-1]
            cube['D'][0] = left_col
            cube['L'][:, 2] = up_row[::-1]
        else:
            cube['B'] = np.rot90(cube['B'], 1)
            up_row = cube['U'][2].copy()
            right_col = cube['R'][:, 0].copy()
            down_row = cube['D'][0].copy()
            left_col = cube['L'][:, 2].copy()
            cube['U'][2] = left_col[::-1]
            cube['R'][:, 0] = up_row
            cube['D'][0] = right_col[::-1]
            cube['L'][:, 2] = down_row
    elif face == 'L':
        if direction == 1:
            cube['L'] = np.rot90(cube['L'], 1)
            up_col = cube['U'][:, 0].copy()
            front_col = cube['F'][:, 0].copy()
            down_col = cube['D'][:, 0].copy()
            back_col = cube['B'][:, 2].copy()
            cube['U'][:, 0] = back_col
            cube['F'][:, 0] = up_col[::-1]
            cube['D'][:, 0] = front_col[::-1]
            cube['B'][:, 2] = down_col
        else:
            cube['L'] = np.rot90(cube['L'], -1)
            up_col = cube['U'][:, 0].copy()
            front_col = cube['F'][:, 0].copy()
            down_col = cube['D'][:, 0].copy()
            back_col = cube['B'][:, 2].copy()
            cube['U'][:, 0] = front_col
            cube['F'][:, 0] = down_col[::-1]
            cube['D'][:, 0] = back_col[::-1]
            cube['B'][:, 2] = up_col
    elif face == 'R':
        if direction == 1:
            cube['R'] = np.rot90(cube['R'], 1)
            up_col = cube['U'][:, 2].copy()
            back_col = cube['B'][:, 0].copy()
            down_col = cube['D'][:, 2].copy()
            front_col = cube['F'][:, 2].copy()
            cube['U'][:, 2] = front_col[::-1]
            cube['B'][:, 0] = up_col
            cube['D'][:, 2] = back_col
            cube['F'][:, 2] = down_col[::-1]
        else:
            cube['R'] = np.rot90(cube['R'], -1)
            up_col = cube['U'][:, 2].copy()
            back_col = cube['B'][:, 0].copy()
            down_col = cube['D'][:, 2].copy()
            front_col = cube['F'][:, 2].copy()
            cube['U'][:, 2] = back_col
            cube['B'][:, 0] = down_col
            cube['D'][:, 2] = front_col[::-1]
            cube['F'][:, 2] = up_col[::-1]

def rotate_middle_horizontal(direction):
    global cube
    if direction == 1:  # Right
        front = cube['F'][1].copy()
        right = cube['R'][1].copy()
        back = cube['B'][1].copy()
        left = cube['L'][1].copy()
        cube['F'][1] = left[::-1]
        cube['R'][1] = front[::-1]
        cube['B'][1] = right[::-1]
        cube['L'][1] = back[::-1]
    else:  # Left
        front = cube['F'][1].copy()
        right = cube['R'][1].copy()
        back = cube['B'][1].copy()
        left = cube['L'][1].copy()
        cube['F'][1] = right[::-1]
        cube['R'][1] = back[::-1]
        cube['B'][1] = left[::-1]
        cube['L'][1] = front[::-1]

# def rotate_middle_vertical(direction):
#     global cube
#     if direction == 1:  # Clockwise (Down)
#         up = cube['U'][1].copy()
#         front = cube['F'][:, 1].copy()
#         down = cube['D'][1].copy()
#         back = cube['B'][:, 1].copy()
#         cube['U'][1] = front
#         cube['F'][:, 1] = down[::-1]
#         cube['D'][1] = back
#         cube['B'][:, 1] = up[::-1]
#     else:  # Counterclockwise (Up)
#         up = cube['U'][1].copy()
#         front = cube['F'][:, 1].copy()
#         down = cube['D'][1].copy()
#         back = cube['B'][:, 1].copy()
#         cube['U'][1] = back[::-1]
#         cube['F'][:, 1] = up
#         cube['D'][1] = front[::-1]
#         cube['B'][:, 1] = down
def rotate_middle_vertical(direction):
    global cube
    temp_col = cube['F'][:, 1][::-1].copy()  # Start with Front middle column, reversed
    if direction == -1:  # Clockwise: Front → Down → Back → Up
        cube['F'][:, 1] = cube['D'][:, 1][::-1]  # Down to Front, reversed
        cube['D'][:, 1] = cube['B'][:, 1]        # Back to Down
        cube['B'][:, 1] = cube['U'][:, 1]        # Up to Back
        cube['U'][:, 1] = temp_col              # Front to Up
    else:  # Counterclockwise: Front → Up → Back → Down
        cube['F'][:, 1] = cube['U'][:, 1][::-1]  # Up to Front, reversed
        cube['U'][:, 1] = cube['B'][:, 1]        # Back to Up
        cube['B'][:, 1] = cube['D'][:, 1]        # Down to Back
        cube['D'][:, 1] = temp_col               # Front to Down

def get_cube_string():
    face_order = ['U', 'R', 'F', 'D', 'L', 'B']
    color_map = {'W': 'U', 'G': 'R', 'R': 'F', 'Y': 'D', 'B': 'L', 'O': 'B'}
    result = ''
    # Create a copy to avoid modifying original cube
    temp_cube = {face: cube[face].copy() for face in face_order}
    # Fix orientations
    temp_cube['U'] = temp_cube['U'][::-1]  # Reverse rows
    temp_cube['D'] = temp_cube['D'][::-1]  # Reverse rows
    temp_cube['R'] = temp_cube['R'][:, ::-1]  # Reverse columns
    temp_cube['L'] = temp_cube['L'][:, ::-1]  # Reverse columns
    for face in face_order:
        for row in temp_cube[face]:
            for color in row:
                result += color_map[color]
    return result

# Define color_map globally (if not already defined)
color_map = {'W': 'U', 'G': 'R', 'R': 'F', 'Y': 'D', 'B': 'L', 'O': 'B'}

# Debug function to print faces and cube string
def debug_cube():
    print("=== Cube Faces (NumPy Arrays) ===")
    for face in ['U', 'R', 'F', 'D', 'L', 'B']:
        print(f"\n{face} Face:")
        for row in cube[face]:
            print(' '.join(row))
        face_string = ''
        for row in cube[face]:
            for color in row:
                face_string += color_map[color]
        face_string_inv = ''
        for row in cube[face][::-1]:
            for color in row:
                face_string_inv += color_map[color]
        print(f"Cube String for {face}: {face_string}")
        print(f"Inverted String for {face}: {face_string_inv}")
        if face_string != face_string_inv:
            print(f"Warning: {face} may be inverted!")

    print("\n=== Full Cube String ===")
    cube_string = get_cube_string()
    print(f"Full String: {cube_string}")
    print(f"Length: {len(cube_string)}")

    print("\n=== U Face Check ===")
    print(f"Expected U String: BBBUUUUUU")
    print(f"Actual U String:   {cube_string[:9]}")

def start_solving():
    global solving, solution_steps, current_step, expected_face, cube
    try:
        cube_string = get_cube_string()
        print(f"Cube String: {cube_string}")
        solution = kociemba.solve(cube_string)
        solution_steps = solution.split()
        print(f"Solution Steps: {solution_steps}")
        if not solution:
            print("Cube is already solved")
            solving = False
            solution_steps = []
            current_step = 0
            expected_face = None
            return
        current_step = 0
        solving = True
        move = solution_steps[0]
        face = move[0]
        direction = 1 if len(move) == 1 else (-1 if move[1] == "'" else 2)
        temp_cube = {f: cube[f].copy() for f in cube}
        rotate_face_3d(face, direction)
        if direction == 2:
            rotate_face_3d(face, direction)
        expected_face = cube['F'].copy() if face in ['F', 'R', 'L', 'B'] else None
        cube = temp_cube
    except Exception as e:
        print(f"Solving failed: {e}")
        debug_cube()
        solving = False
        solution_steps = []
        current_step = 0
        expected_face = None

def process_gesture(gesture_text):
    global last_gesture, solving, current_step, solution_steps, expected_face, cube
    if gesture_text == "Reset Cube":
        reset_cube()
    elif not solving:
        if gesture_text == "Rotate Right Face":
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
            rotate_middle_vertical(-1)
        elif gesture_text == "Middle Column Down":
            rotate_middle_vertical(1)
    elif solving and current_step < len(solution_steps):
        move = solution_steps[current_step]
        face = move[0]
        direction = 1 if len(move) == 1 else (-1 if move[1] == "'" else 2)
        gesture_map = {
            "R": "Rotate Right Face",
            "U": "Rotate Up Face",
            "F": "Rotate Front Face",
            "L": "Rotate Left Face",
            "D": "Rotate Down Face",
            "B": "Rotate Back Face"
        }
        expected_gesture = gesture_map.get(face, "")
        if gesture_text == expected_gesture:
            print(f"Applying {gesture_text} for move {move}")
            if direction == 1:
                rotate_face_3d(face, 1)
            elif direction == -1:
                rotate_face_3d(face, 1)
                print("Completing counterclockwise: Applying two additional clockwise rotations")
                rotate_face_3d(face, 1)
                rotate_face_3d(face, 1)
            elif direction == 2:
                rotate_face_3d(face, 1)
                print("Applying second 90° for 180° move")
                rotate_face_3d(face, 1)
            current_step += 1
            if current_step < len(solution_steps):
                move = solution_steps[current_step]
                face = move[0]
                direction = 1 if len(move) == 1 else (-1 if move[1] == "'" else 2)
                temp_cube = {f: cube[f].copy() for f in cube}
                rotate_face_3d(face, direction)
                if direction == 2:
                    rotate_face_3d(face, direction)
                expected_face = cube['F'].copy() if face in ['F', 'R', 'L', 'B'] else None
                cube = temp_cube
            else:
                solving = False
                solution_steps = []
                expected_face = None
                print("Solving complete")
    last_gesture = gesture_text

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

    def is_ring_up(hand):
        return hand.landmark[16].y < hand.landmark[14].y  # Ring tip above PIP

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
        norm_product = np.linalg.norm(palm_vector) * np.linalg.norm (index_vector)
        if norm_product == 0:
            return False
        angle = math.degrees(math.acos(np.clip(dot / norm_product, -1.0, 1.0)))
        return angle < 45

    # Both hands gestures
    if left and right:
        if is_palm_open(left) and is_palm_open(right):
            return "Reset Cube"
        # if is_index_up(right) and is_fist(left):
        #     return "Middle Row Right"
        # if is_index_up(left) and is_fist(right):
        #     return "Middle Row Left"

    # Right hand gestures
    if right:
        if is_index_up(right):
            return "Rotate Right Face"
        if is_pinky_up(right):
            return "Rotate Up Face"
        # if is_middle_up(right):
        #     return "Middle Column Up"
        if is_ring_up(right):
            return "Rotate Front Face"

    # Left hand gestures
    if left:
        if is_index_up(left):
            return "Rotate Left Face"
        if is_pinky_up(left):
            return "Rotate Down Face"
        # if is_middle_up(left):
        #     return "Middle Column Down"
        if is_ring_up(left):
            return "Rotate Back Face"

    return ""

# Main loop
waiting_for_rest = False
running = True
init()
dragging = False
last_mouse_pos = (0, 0)
x_rot, y_rot = 35.0, 45.0
sensitivity = 0.5

while running:
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -10)
    glRotatef(x_rot, 1, 0, 0)
    glRotatef(y_rot, 0, 1, 0)
    draw_full_cube()

    button_x, button_y, button_width, button_height = draw_button()

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

    if waiting_for_rest:
        if gesture_text == "":
            waiting_for_rest = False
            last_gesture = ""
    elif gesture_text and gesture_text != last_gesture:
        process_gesture(gesture_text)
        waiting_for_rest = True

    status_text = f"Gesture: {gesture_text}"
    if solving and current_step < len(solution_steps):
        status_text += f" | Next Move: {solution_steps[current_step]}"
        remaining_steps = "Steps: " + " ".join(solution_steps[current_step:])
        cv2.putText(frame, remaining_steps, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Hand Tracking", frame)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                x, y = event.pos
                if (button_x <= x <= button_x + button_width and 
                    display[1] - (button_y + button_height) <= y <= display[1] - button_y):
                    if not solving:
                        print("Button clicked, starting solve")
                        start_solving()
                else:
                    dragging = True
                    last_mouse_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if dragging:
                x, y = event.pos
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