import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

colors = {
    'W': (1, 1, 1),     # White
    'Y': (1, 1, 0),     # Yellow
    'R': (1, 0, 0),     # Red
    'O': (1, 0.5, 0),   # Orange
    'B': (0, 0, 1),     # Blue
    'G': (0, 1, 0),     # Green
}

cube = {
    'U': np.full((3, 3), 'W'),
    'D': np.full((3, 3), 'Y'),
    'F': np.full((3, 3), 'R'),
    'B': np.full((3, 3), 'O'),
    'L': np.full((3, 3), 'B'),
    'R': np.full((3, 3), 'G'),
}

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
        'L': ((-1, 0, 0),  (0, -90, 0)),   # Left (adjusted rotation)
        'R': ((1, 0, 0),   (0, 90, 0)),    # Right (adjusted rotation)
    }

    glPushMatrix()
    glTranslatef(x, y, z)

    # Draw cubelet body (dark cube) - temporarily disabled for debugging
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
        glVertex3f(-half * 0.9, -half * 0.9, 0.05)  # Increased offset
        glVertex3f( half * 0.9, -half * 0.9, 0.05)
        glVertex3f( half * 0.9,  half * 0.9, 0.05)
        glVertex3f(-half * 0.9,  half * 0.9, 0.05)
        glEnd()
        glPopMatrix()

    glPopMatrix()

def draw_full_cube():
    spacing = 1.1
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if x in [0, 2] or y in [0, 2] or z in [0, 2]:
                    stickers = {}
                    if z == 2: stickers['F'] = cube['F'][2-y][x]    # Red
                    if z == 0: stickers['B'] = cube['B'][2-y][2-x]  # Orange
                    if y == 2: stickers['U'] = cube['U'][2-z][x]    # White
                    if y == 0: stickers['D'] = cube['D'][z][x]      # Yellow
                    if x == 0: stickers['L'] = cube['L'][2-y][2-z]  # Blue
                    if x == 2: stickers['R'] = cube['R'][2-y][z]    # Green
                    draw_cubelet((x-1)*spacing, (y-1)*spacing, (z-1)*spacing, 0.9, stickers)

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)
    gluPerspective(45, (800/600), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -10)
    glEnable(GL_DEPTH_TEST)
    glDisable(GL_CULL_FACE)  # Disable face culling to ensure all faces are visible

    clock = pygame.time.Clock()
    x_rot, y_rot = 35, 45  # Adjusted for isometric view
    dragging = False
    last_mouse_pos = (0, 0)

    running = True
    while running:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

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
                    y_rot += dx * 0.5
                    x_rot += dy * 0.5
                    last_mouse_pos = (x, y)

        glPushMatrix()
        glRotatef(x_rot, 1, 0, 0)
        glRotatef(y_rot, 0, 1, 0)
        draw_full_cube()
        glPopMatrix()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()