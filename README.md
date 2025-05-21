# Gesture-Controlled Rubik’s Cube Solver

A Python-based system to solve a 3x3 Rubik’s Cube using hand gestures detected via webcam. This project leverages MediaPipe for real-time gesture recognition, OpenCV for visualization, and the Kociemba algorithm to generate optimal solving sequences. Users can scramble and solve the cube interactively by performing gestures (e.g., raising the right index finger to rotate the right face).

## Features

- **Gesture Control**: Perform cube rotations using intuitive hand gestures detected via webcam.
- **Automatic Solving**: Uses the Kociemba algorithm to generate efficient solving sequences.
- **Real-Time Visualization**: Displays the cube state using OpenCV windows.
- **Robust State Tracking**: Manages cube state with NumPy arrays, correcting orientation mismatches.
- **Counterclockwise Support**: Converts counterclockwise moves (e.g., R') into multiple clockwise rotations for seamless interaction.

## Technologies Used

- **Python 3.x**
- **OpenCV**: For webcam input and cube visualization.
- **MediaPipe**: For real-time hand gesture detection.
- **NumPy**: For cube state management.
- **Kociemba**: For generating cube-solving sequences.

## Prerequisites

- Python 3.8 or higher
- A webcam for gesture detection
- Required Python libraries (listed in `requirements.txt`)

## Installation

1. **Clone the Repository** 

   ```bash
   git clone https://github.com/noughtsad/Rubiks-Cube.git
   cd rubiks-cube-solver
   ```

2. **Set Up a Virtual Environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r Requirements.txt
   ```

4. **Verify Webcam Access**:

   - Ensure your webcam is connected and accessible by Python/OpenCV.

## Usage

1. **Run the Program**:

   ```bash
   python main.py
   ```

2. **Scramble the Cube**:

   - Use gestures to scramble the cube (see Gestures and Actions below for mappings).
   - Example: Raise your right index finger to rotate the right face (R).

3. **Solve the Cube**:

   - Click the “Solve” button in the OpenCV window.
   - Follow the “Next Move” instructions displayed on the webcam feed (e.g., `R`, `R'`, `U'`).
   - Perform the corresponding gesture for each move.
   - Counterclockwise moves (e.g., `R'`, `U'`) are automatically handled by applying three clockwise rotations.

4. **View Results**:

   - The cube will solve step-by-step, aligning all faces (white on top, yellow on bottom, etc.).
   - Console logs provide debugging info (e.g., cube state, applied moves).

## Gestures and Actions

The following hand gestures are used to rotate the cube faces. Each gesture corresponds to a standard Rubik’s Cube move (R, L, U, D, F, B). Ensure your hand is clearly visible to the webcam for accurate detection.

| Gesture | Action | Cube Move |
| --- | --- | --- |
| Right index finger up | Rotate Right face clockwise | R |
| Left index finger up | Rotate Left face clockwise | L |
| Right pinky finger up | Rotate Up face clockwise | U |
| Left pinky finger up | Rotate Down face clockwise | D |
| Right ring finger up | Rotate Front face clockwise | F |
| Left ring finger up | Rotate Back face clockwise | B |

**Notes**:

- To perform a counterclockwise move (e.g., `R'`, `U'`), the system applies three clockwise rotations of the same face (e.g., `R + R + R = R'`).
- For 180-degree moves (e.g., `R2`), the system applies two clockwise rotations (e.g., `R + R`).

## Project Structure

- `main.py`: Main script containing gesture detection, cube logic, and solver integration.
- `Requirements.txt`: List of Python dependencies.
- `README.md`: Project documentation (this file).
- `LICENSE`: MIT License file.
- `2dcube_working.py`: Used initially for understanding.
- `3dworking.py`: Used initially for understanding.

## Contributing

Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your changes. For major updates, open an issue to discuss first.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or feedback, reach out via GitHub: https://github.com/noughtsad
