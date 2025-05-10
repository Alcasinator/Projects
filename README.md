Gesture-Controlled Rubik’s Cube Solver
A Python-based system to solve a 3x3 Rubik’s Cube using hand gestures detected via webcam. This project leverages MediaPipe for real-time gesture recognition, OpenCV for visualization, and the Kociemba algorithm to generate optimal solving sequences. Users can scramble and solve the cube interactively by performing gestures (e.g., raising the right index finger to rotate the right face).
Features

Gesture Control: Perform cube rotations using hand gestures (e.g., right index for right face, pinky for up face).
Automatic Solving: Uses the Kociemba algorithm to solve the cube efficiently.
Real-Time Visualization: Displays the cube state using OpenCV.
Robust State Tracking: Handles cube state with NumPy arrays, correcting orientation mismatches.
Counterclockwise Support: Converts counterclockwise moves (e.g., R') into multiple clockwise rotations for seamless user interaction.

Technologies Used

Python 3.x
OpenCV: For webcam input and cube visualization.
MediaPipe: For hand gesture detection.
NumPy: For cube state management.
Kociemba: For generating cube-solving sequences.

Prerequisites

Python 3.8 or higher
A webcam for gesture detection
Required Python libraries (listed in requirements.txt)

Installation

Clone the Repository :git clone https://github.com/noughtsad/Rubiks-Cube.git

cd rubiks-cube-solver


Set Up a Virtual Environment (optional but recommended):python -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies:pip install -r requirements.txt

Verify Webcam Access:
Ensure your webcam is connected and accessible by Python/OpenCV.

Usage
Run the Program:python solving.py

Scramble the Cube:
Use gestures to scramble the cube:
Right index finger: Rotate right face (R).
Right pinky: Rotate up face (U).
See console for other gesture mappings.

Solve the Cube:
Click the green button in the OpenCV window.
Follow the “Next Move” instructions displayed on the webcam feed (e.g., R, R', U').
Perform the corresponding gesture (e.g., right index for R or R').
The program automatically converts counterclockwise moves (e.g., R') into three clockwise rotations.


View Results:
The cube will solve step-by-step, aligning all faces (white on top, yellow on bottom, etc.).


Contributing
Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your changes. For major updates, open an issue to discuss first.
Contact
For questions or feedback, reach out via GitHub: noughtsad.
