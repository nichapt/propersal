# Face Mesh Project

This project uses OpenCV and MediaPipe to detect facial landmarks, estimate head direction, and determine eye states (open/closed) in real-time using your webcam.

## Features
- Real-time face mesh and iris detection
- Head direction estimation (left, right, forward)
- Eye state detection (open, left closed, right closed, both closed)
- Visual overlay of results on webcam feed

## Setup
1. **Create and activate a virtual environment:**
   - Run the provided PowerShell script:
     ```powershell
     .\create_venv.ps1
     ```
2. **Install dependencies:**
   - After activating the environment, run:
     ```powershell
     pip install -r requirements.txt
     ```

## Usage
- Run the main script:
  ```powershell
  python mdp_cv2.py
  ```
- Press `ESC` to exit the webcam window.

## Files
- `mdp_cv2.py`: Main script for face mesh and eye state detection
- `requirements.txt`: Python dependencies
- `create_venv.ps1`: PowerShell script to set up virtual environment
- `.gitignore`: Recommended files to ignore in version control

## Notes
- Make sure your webcam is connected and accessible.
- For best compatibility, use Python 3.10 or 3.11.
- If you encounter MediaPipe errors, check your Python version and package installation.

## License
This project is for educational and research purposes.
