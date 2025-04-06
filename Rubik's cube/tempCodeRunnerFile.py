import pygame
import numpy as np

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((600, 600))
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

def draw_face(face, x_offset, y_offset):
    for i in range(3):
        for j in range(3):
            color = colors[cube[face][i][j]]
            pygame.draw.rect(screen, color, (x_offset + j*50, y_offset + i*50, 48, 48))
            pygame.draw.rect(screen, (0, 0, 0), (x_offset + j*50, y_offset + i*50, 48, 48), 2)

def draw_cube():
    screen.fill((30, 30, 30))
    draw_face("U", 150, 50)
    draw_face("L", 50, 200)
    draw_face("F", 150, 200)
    draw_face("R", 250, 200)
    draw_face("B", 350, 200)
    draw_face("D", 150, 350)
    pygame.display.flip()

def rotate_front_clockwise():
    global cube
    cube["F"] = np.rot90(cube["F"], -1)  # Rotate the front face

    # Store the edges before rotating
    top_row = cube["U"][2].copy()
    left_col = cube["L"][:, 2].copy()
    bottom_row = cube["D"][0].copy()
    right_col = cube["R"][:, 0].copy()

    # Rotate edges correctly
    cube["U"][2] = left_col[::-1]  # Left column (inverted) moves to top
    cube["L"][:, 2] = bottom_row  # Bottom row moves to left column
    cube["D"][0] = right_col[::-1]  # Right column (inverted) moves to bottom row
    cube["R"][:, 0] = top_row  # Top row moves to right column


# Main loop
running = True
while running:
    draw_cube()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_f:  # Rotate front clockwise
                rotate_front_clockwise()

    clock.tick(30)

pygame.quit()
