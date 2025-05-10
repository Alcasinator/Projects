import numpy as np

# Initialize cube
cube = {
    'U': np.array([['O', 'O', 'O'], ['W', 'W', 'W'], ['W', 'W', 'W']]),
    'D': np.full((3, 3), 'Y'),
    'F': np.full((3, 3), 'R'),
    'B': np.full((3, 3), 'O'),
    'L': np.full((3, 3), 'B'),
    'R': np.full((3, 3), 'G'),
}

# Define get_cube_string()
def get_cube_string():
    face_order = ['U', 'R', 'F', 'D', 'L', 'B']
    color_map = {'W': 'U', 'G': 'R', 'R': 'F', 'Y': 'D', 'B': 'L', 'O': 'B'}
    result = ''
    for face in face_order:
        for row in cube[face]:
            for color in row:
                result += color_map[color]
    return result

# Debug function
color_map = {'W': 'U', 'G': 'R', 'R': 'F', 'Y': 'D', 'B': 'L', 'O': 'B'}
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

# Run debug
debug_cube()