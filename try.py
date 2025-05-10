import pygame
import sys
from pygame.locals import *

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 1000, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rubik's Cube 2D Visualization")
font = pygame.font.SysFont('Arial', 24)
clock = pygame.time.Clock()

# Cube colors (RGB)
COLORS = {
    'U': (255, 255, 255),  # White (Up)
    'D': (255, 255, 0),    # Yellow (Down)
    'L': (0, 255, 0),      # Green (Left)
    'R': (0, 0, 255),      # Blue (Right)
    'F': (255, 0, 0),      # Red (Front)
    'B': (255, 165, 0)     # Orange (Back)
}

class Cube:
    def __init__(self):
        self.faces = {f: [[f]*3 for _ in range(3)] for f in COLORS}

    def rotate_face(self, face, clockwise=True):
        rotated = [list(row) for row in zip(*self.faces[face][::-1])] if clockwise else \
                 [list(row) for row in zip(*self.faces[face])][::-1]
        self.faces[face] = rotated

    def rotate_U(self, clockwise=True):
        self.rotate_face('U', clockwise)
        temp = self.faces['F'][0].copy()
        if clockwise:
            self.faces['F'][0] = self.faces['R'][0]
            self.faces['R'][0] = self.faces['B'][0]
            self.faces['B'][0] = self.faces['L'][0]
            self.faces['L'][0] = temp
        else:
            self.faces['F'][0] = self.faces['L'][0]
            self.faces['L'][0] = self.faces['B'][0]
            self.faces['B'][0] = self.faces['R'][0]
            self.faces['R'][0] = temp

    # Other rotation methods remain the same as previous example
    # (D, L, R, F, B rotations would go here)

def draw_cube(cube):
    screen.fill((50, 50, 50))
    face_size = 60
    spacing = 15
    border = 2
    
    # Calculate center positions
    center_x, center_y = WIDTH//2, HEIGHT//2
    
    # Draw Up face (centered above middle)
    up_x = center_x - 1.5*face_size - spacing
    up_y = center_y - 3*face_size - 2*spacing
    for i in range(3):
        for j in range(3):
            color = COLORS[cube.faces['U'][i][j]]
            pygame.draw.rect(screen, color, 
                           (up_x + j*(face_size+spacing),
                           up_y + i*(face_size+spacing),
                           face_size, face_size))
            pygame.draw.rect(screen, (0,0,0), 
                           (up_x + j*(face_size+spacing),
                           up_y + i*(face_size+spacing),
                           face_size, face_size), border)
    
    # Draw middle faces (Left, Front, Right, Back)
    # Left face
    left_x = center_x - 3*(face_size+spacing)
    left_y = center_y - 1.5*face_size - spacing
    for i in range(3):
        for j in range(3):
            color = COLORS[cube.faces['L'][i][j]]
            pygame.draw.rect(screen, color,
                           (left_x + j*(face_size+spacing),
                           left_y + i*(face_size+spacing),
                           face_size, face_size))
            pygame.draw.rect(screen, (0,0,0),
                           (left_x + j*(face_size+spacing),
                           left_y + i*(face_size+spacing),
                           face_size, face_size), border)
    
    # Front face (center)
    front_x = center_x - 1.5*face_size - spacing
    front_y = center_y - 1.5*face_size - spacing
    for i in range(3):
        for j in range(3):
            color = COLORS[cube.faces['F'][i][j]]
            pygame.draw.rect(screen, color,
                           (front_x + j*(face_size+spacing),
                           front_y + i*(face_size+spacing),
                           face_size, face_size))
            pygame.draw.rect(screen, (0,0,0),
                           (front_x + j*(face_size+spacing),
                           front_y + i*(face_size+spacing),
                           face_size, face_size), border)
    
    # Right face
    right_x = center_x + 1.5*face_size + spacing
    right_y = center_y - 1.5*face_size - spacing
    for i in range(3):
        for j in range(3):
            color = COLORS[cube.faces['R'][i][j]]
            pygame.draw.rect(screen, color,
                           (right_x + j*(face_size+spacing),
                           right_y + i*(face_size+spacing),
                           face_size, face_size))
            pygame.draw.rect(screen, (0,0,0),
                           (right_x + j*(face_size+spacing),
                           right_y + i*(face_size+spacing),
                           face_size, face_size), border)
    
    # Back face
    back_x = center_x + 4.5*face_size + 3*spacing
    back_y = center_y - 1.5*face_size - spacing
    for i in range(3):
        for j in range(3):
            color = COLORS[cube.faces['B'][i][j]]
            pygame.draw.rect(screen, color,
                           (back_x + j*(face_size+spacing),
                           back_y + i*(face_size+spacing),
                           face_size, face_size))
            pygame.draw.rect(screen, (0,0,0),
                           (back_x + j*(face_size+spacing),
                           back_y + i*(face_size+spacing),
                           face_size, face_size), border)
    
    # Draw Down face (centered below middle)
    down_x = center_x - 1.5*face_size - spacing
    down_y = center_y + 1.5*face_size + 2*spacing
    for i in range(3):
        for j in range(3):
            color = COLORS[cube.faces['D'][i][j]]
            pygame.draw.rect(screen, color,
                           (down_x + j*(face_size+spacing),
                           down_y + i*(face_size+spacing),
                           face_size, face_size))
            pygame.draw.rect(screen, (0,0,0),
                           (down_x + j*(face_size+spacing),
                           down_y + i*(face_size+spacing),
                           face_size, face_size), border)
    
    # Draw instructions
    instructions = [
        "Press keys to rotate: U/D/L/R/F/B - Clockwise",
        "Hold Shift for counter-clockwise",
        "ESC to quit"
    ]
    for i, line in enumerate(instructions):
        text = font.render(line, True, (255, 255, 255))
        screen.blit(text, (20, 20 + i*30))

def main():
    cube = Cube()
    
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == KEYDOWN:
                shift_pressed = pygame.key.get_mods() & KMOD_SHIFT
                
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                
                elif event.key == K_u:
                    cube.rotate_U(not shift_pressed)
                elif event.key == K_d:
                    cube.rotate_D(not shift_pressed)
                elif event.key == K_l:
                    cube.rotate_L(not shift_pressed)
                elif event.key == K_r:
                    cube.rotate_R(not shift_pressed)
                elif event.key == K_f:
                    cube.rotate_F(not shift_pressed)
                elif event.key == K_b:
                    cube.rotate_B(not shift_pressed)
        
        draw_cube(cube)
        pygame.display.flip()
        clock.tick(30)

if __name__ == "__main__":
    main()