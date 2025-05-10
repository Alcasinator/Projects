import pygame
import numpy as np

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
def create_cube():
    return {
        "U": np.full((3, 3), "W"),
        "D": np.full((3, 3), "Y"),
        "F": np.full((3, 3), "R"),
        "B": np.full((3, 3), "O"),
        "L": np.full((3, 3), "B"),
        "R": np.full((3, 3), "G"),
    }

cube = create_cube()

# Draw one face
def draw_face(face, x_offset, y_offset):
    for i in range(3):
        for j in range(3):
            color = colors[cube[face][i][j]]
            pygame.draw.rect(screen, color, (x_offset + j * 50, y_offset + i * 50, 48, 48))
            pygame.draw.rect(screen, (0, 0, 0), (x_offset + j * 50, y_offset + i * 50, 48, 48), 2)

# Draw full cube
def draw_cube():
    screen.fill((30, 30, 30))
    draw_face("U", 200, 50)
    draw_face("L", 50, 200)
    draw_face("F", 200, 200)
    draw_face("R", 350, 200)
    draw_face("B", 500, 200)
    draw_face("D", 200, 350)
    pygame.display.flip()

# Reset cube
def reset_cube():
    global cube
    cube = create_cube()

# Generic face rotator
def rotate_face(face, direction):
    cube[face] = np.rot90(cube[face], -direction)

# Front face rotation
def rotate_front(direction):
    rotate_face("F", direction)
    for _ in range((direction + 4) % 4):
        top = cube["U"][2].copy()
        left = cube["L"][:, 2].copy()
        bottom = cube["D"][0].copy()
        right = cube["R"][:, 0].copy()

        cube["U"][2] = left[::-1]
        cube["L"][:, 2] = bottom
        cube["D"][0] = right[::-1]
        cube["R"][:, 0] = top

# Right face rotation
def rotate_right(direction):
    rotate_face("R", direction)
    for _ in range((direction + 4) % 4):
        top = cube["U"][:, 2].copy()
        front = cube["F"][:, 2].copy()
        bottom = cube["D"][:, 2].copy()
        back = cube["B"][:, 0].copy()

        cube["U"][:, 2] = front
        cube["F"][:, 2] = bottom
        cube["D"][:, 2] = back[::-1]
        cube["B"][:, 0] = top[::-1]

# Left face rotation
def rotate_left(direction):
    rotate_face("L", direction)
    for _ in range((direction + 4) % 4):
        top = cube["U"][:, 0].copy()
        front = cube["F"][:, 0].copy()
        bottom = cube["D"][:, 0].copy()
        back = cube["B"][:, 2].copy()

        cube["U"][:, 0] = back[::-1]
        cube["F"][:, 0] = top
        cube["D"][:, 0] = front
        cube["B"][:, 2] = bottom[::-1]

# Up face rotation
def rotate_up(direction):
    rotate_face("U", direction)
    for _ in range((direction + 4) % 4):
        front = cube["F"][0].copy()
        right = cube["R"][0].copy()
        back = cube["B"][0].copy()
        left = cube["L"][0].copy()

        cube["F"][0] = right
        cube["R"][0] = back
        cube["B"][0] = left
        cube["L"][0] = front

# Down face rotation
def rotate_down(direction):
    rotate_face("D", direction)
    for _ in range((direction + 4) % 4):
        front = cube["F"][2].copy()
        right = cube["R"][2].copy()
        back = cube["B"][2].copy()
        left = cube["L"][2].copy()

        cube["F"][2] = left
        cube["R"][2] = front
        cube["B"][2] = right
        cube["L"][2] = back

# Back face rotation
def rotate_back(direction):
    rotate_face("B", direction)
    for _ in range((direction + 4) % 4):
        top = cube["U"][0].copy()
        left = cube["L"][:, 0].copy()
        bottom = cube["D"][2].copy()
        right = cube["R"][:, 2].copy()

        cube["U"][0] = right[::-1]
        cube["R"][:, 2] = bottom
        cube["D"][2] = left[::-1]
        cube["L"][:, 0] = top

# Main loop
running = True
while running:
    draw_cube()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            key = event.key
            if key == pygame.K_f: rotate_front(1)
            elif key == pygame.K_v: rotate_front(-1)
            elif key == pygame.K_r: rotate_right(1)
            elif key == pygame.K_k: rotate_right(-1)
            elif key == pygame.K_l: rotate_left(1)
            elif key == pygame.K_a: rotate_left(-1)
            elif key == pygame.K_u: rotate_up(1)
            elif key == pygame.K_j: rotate_up(-1)
            elif key == pygame.K_d: rotate_down(1)
            elif key == pygame.K_m: rotate_down(-1)
            elif key == pygame.K_b: rotate_back(1)
            elif key == pygame.K_n: rotate_back(-1)
            elif key == pygame.K_z: reset_cube()

    clock.tick(30)

pygame.quit()
